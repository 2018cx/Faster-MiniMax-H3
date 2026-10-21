"""Independently audit completed raw trajectory records; never run training.

This checks recorded contracts and recomputes the original endpoint rule. It
does not inspect checkpoint tensor contents or replace the full terminal audit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from fractions import Fraction
import statistics

SEEDS = (424242, 424243, 424244)
METRICS = ('total_loss', 'video_loss', 'audio_loss', 'gradient_l2')
ORDER = ((424242, 'reference'), (424242, 'candidate'),
         (424243, 'candidate'), (424243, 'reference'),
         (424244, 'reference'), (424244, 'candidate'))
PAIRED_FIELDS = ('step', 'rng_before', 'rng_after_noise', 'rng_after', 'item',
                 'layout', 'sigma_v', 'sigma_a', 'model_inputs', 'audio_weight')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def mean(values):
    # Exact binary rational sum, rounded only when converting the final mean.
    return float(sum((Fraction.from_float(x) for x in values), Fraction()) / len(values))


def audit(directory):
    manifest = load(directory / 'manifest.json')
    summary = load(directory / 'summary.json')
    require(summary.get('complete') and summary.get('status') in ('PASS', 'FAIL'),
            'Only terminal, complete scientific results can be audited')
    require(summary['seeds'] == list(SEEDS) and summary['optimizer_steps'] == 50,
            'Original seeds and exactly50 updates required')
    require(len(manifest['arms']) == len(summary['arms']) == 6, 'Six fresh arms required')
    require(manifest['arms'] == summary['arms'], 'Manifest/summary arm records differ')
    require([(a['seed'], a['label']) for a in manifest['arms']] == list(ORDER),
            'Balanced execution order changed')
    observer_sha = digest(directory / 'validate.py')
    require(observer_sha == manifest['instrumentation_sha256'] == summary['instrumentation_sha256'],
            'Observer digest mismatch')
    require(summary['selected_commit'] == manifest['selection']['commit'], 'Selected commit mismatch')
    data = {}
    endpoints = {}
    record_count = 0
    fingerprint = None
    raw_files = {}
    for arm_info, (seed, label) in zip(manifest['arms'], ORDER):
        arm = directory / str(seed) / label
        require(load(arm / 'arm.json') == arm_info, 'Arm metadata differs from terminal manifest')
        require(arm_info['status'] == 'PASS' and arm_info['complete'], 'Incomplete arm')
        require(arm_info['inputs_before']['ok'] and arm_info['inputs_after']['ok'], 'Frozen inputs differ')
        require(arm_info['source_unchanged'], 'Source mutation reported')
        require(arm_info['source_digest'] == manifest['source_baselines'][label], 'Source baseline mismatch')
        require(arm_info['fingerprint_before'] == arm_info['fingerprint_after'], 'Fingerprint changed')
        fp = dict(arm_info['fingerprint_before'])
        require(fp.pop('seed') == seed and fp['steps'] == 50 and fp['world_size'] == 8,
                'Seed, update count or world size changed')
        require(fp['instrumentation_sha256'] == observer_sha, 'Arm observer fingerprint differs')
        if fingerprint is None:
            fingerprint = fp
        require(fp == fingerprint, 'Non-seed hardware/software/instrumentation fingerprint differs')
        cleanup = load(arm / 'compiler-cleanup.json')
        require(cleanup['files_left'] == 0 and not (arm / 'compiler-cache').exists(), 'Cache cleanup incomplete')
        require(digest(arm / 'output/checkpoint-50.pt') == arm_info['checkpoint_sha256'],
                'Checkpoint bytes differ (tensor contents require separate audit)')
        log = (arm / 'launcher.log').read_text()
        require('LEN_SKIP' not in log and 'SAVE_SKIPPED' not in log, 'Skipped workload')
        ranks = []
        for rank in range(8):
            output = arm / 'output'
            paths = [output / f'initial-rank-{rank}.json',
                     output / f'complete-rank-{rank}.json',
                     output / f'records-rank-{rank}.jsonl']
            initial, done = load(paths[0]), load(paths[1])
            records = [json.loads(line) for line in paths[2].read_text().splitlines()]
            require(len(initial['trainables']) == 108 and
                    sum(v['numel'] for v in initial['trainables'].values()) == 10655744,
                    'Trainable coverage changed')
            require(done['status'] == 'PASS' and done['steps'] == 50 and
                    done['skipped_samples'] == done['skipped_updates'] == 0 and
                    done['norm_checks'] >= 50 and done['instrumentation_sha256'] == observer_sha,
                    'Completion contract failed')
            require([r['step'] for r in records] == list(range(1, 51)), 'Update sequence changed')
            for row in records:
                require(row['global_steps'] == row['step'] and row['skipped_steps'] == 0 and
                        row['norm_checks'] >= row['step'], 'Applied-update contract failed')
                require(set(row['metrics']) == set(METRICS) and
                        all(math.isfinite(row['metrics'][m]) for m in METRICS) and
                        row['metrics']['gradient_l2'] >= 0, 'Nonfinite or invalid metric')
            record_count += len(records)
            ranks.append((initial, done, records))
            for path in paths:
                raw_files[str(path.relative_to(directory))] = digest(path)
        data[(seed, label)] = ranks
        endpoints[(seed, label)] = {}
        for step in (25, 50):
            values = [r[2][step - 1]['metrics'] for r in ranks]
            require(len({v['gradient_l2'] for v in values}) == 1, 'Global gradient norm differs across ranks')
            endpoints[(seed, label)][str(step)] = {
                metric: values[0][metric] if metric == 'gradient_l2' else
                mean([v[metric] for v in values]) for metric in METRICS}
        require(endpoints[(seed, label)] == arm_info['metrics'], 'Recorded arm endpoints disagree with raw records')
    for seed in SEEDS:
        for reference, candidate in zip(data[(seed, 'reference')], data[(seed, 'candidate')]):
            require(reference[0] == candidate[0], 'Paired initial state/RNG/precision differs')
            require(reference[1]['contract'] == candidate[1]['contract'] and
                    reference[1]['precision'] == candidate[1]['precision'], 'Paired optimizer/precision differs')
            for left, right in zip(reference[2], candidate[2]):
                require(all(left[k] == right[k] for k in PAIRED_FIELDS), 'Paired input/RNG/update contract differs')
        pair = manifest['pairs'][str(seed)]
        require(pair['reference'] == endpoints[(seed, 'reference')] and
                pair['candidate'] == endpoints[(seed, 'candidate')], 'Manifest pair endpoints differ')
    rows = []
    for step in (25, 50):
        for metric in METRICS:
            references = [endpoints[(s, 'reference')][str(step)][metric] for s in SEEDS]
            candidates = [endpoints[(s, 'candidate')][str(step)][metric] for s in SEEDS]
            differences = [c - r for r, c in zip(references, candidates)]
            difference = mean(differences)
            sd = statistics.stdev(references)
            ratio = difference / sd if sd else (0.0 if difference == 0.0 else math.copysign(math.inf, difference))
            row = {'step': step, 'metric': metric, 'seeds': list(SEEDS),
                   'reference_values': references, 'candidate_values': candidates,
                   'paired_differences': differences, 'reference_mean': mean(references),
                   'candidate_mean': mean(candidates), 'reference_sample_sd': sd,
                   'candidate_sample_sd': statistics.stdev(candidates), 'mean_difference': difference,
                   'standardized_mean_difference': ratio if math.isfinite(ratio) else
                   ('Infinity' if ratio > 0 else '-Infinity'),
                   'threshold': 2.0, 'pass': math.isfinite(ratio) and abs(ratio) <= 2.0}
            rows.append(row)
    require(rows == summary['rows'], 'Frozen endpoint decisions differ from independent raw recomputation')
    verdict = 'PASS' if all(r['pass'] for r in rows) else 'FAIL'
    require(verdict == summary['status'] and summary['all_integrity_gates_pass'], 'Frozen summary verdict mismatch')
    return {'status': 'PASS_RECOMPUTATION_AUDIT', 'scope': 'Raw records/contracts/endpoints and checkpoint byte hashes; no tensor-content reload',
            'observed_statistical_status': verdict, 'raw_records_verified': record_count,
            'initial_and_completion_files_each': 48, 'rows': rows, 'raw_file_sha256': raw_files,
            'original_gate_unchanged': True, 'all_scientific_gates_verified': False,
            'remaining_audit': 'Checkpoint contents, runtime FP32 optimizer assertions, current source/ref and publication evidence bindings',
            'summary_sha256': digest(directory / 'summary.json'),
            'manifest_sha256': digest(directory / 'manifest.json'), 'observer_sha256': observer_sha}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('validation_directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Never overwrite an audit result')
    result = audit(args.validation_directory.resolve())
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'observed_statistical_status', 'raw_records_verified')}))
