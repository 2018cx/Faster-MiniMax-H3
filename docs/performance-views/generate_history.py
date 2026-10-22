#!/usr/bin/env python3
"""Rebuild the README table and SVG from checked-in evidence; no GPU required."""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'docs/performance-data/rounds-01-40.json'
SVG = ROOT / 'docs/performance-views/02-history.svg'
BEGIN = '<!-- BEGIN GENERATED OPTIMIZATION HISTORY -->'
END = '<!-- END GENERATED OPTIMIZATION HISTORY -->'
BURGUNDY, INK, MUTED, GRID, SOFT = '#8c0000', '#292122', '#79696b', '#eadfe0', '#faf5f5'


def validate(data):
    rows = data['rounds']
    assert [r['round'] for r in rows] == list(range(1, 41))
    evidence = {e['path']: e for e in data['evidence']}
    best, best_round = 1.0, 0
    for row in rows:
        for name, kind in [('best_screen', 'screening'), ('full_check', 'full_check'), ('profile_observation', 'profile')]:
            p = row[name]
            if p:
                e = evidence[p['path']]
                assert e['kind'] == kind and e['status'] == 'PASS' and e['complete']
                assert e['performance']['candidate_step_seconds'] == p['seconds']
                assert e['performance']['speedup'] == p['speedup']
                assert abs(p['reference_seconds'] / p['seconds'] - p['speedup']) < 1e-12
                if kind == 'screening':
                    assert p['reference_seconds'] == data['scope']['screening_reference_seconds']
        if row['full_check']:
            assert row['full_check']['speedup'] > best
            best, best_round = row['full_check']['speedup'], row['round']
        assert row['accepted_best_speedup'] == best and row['accepted_best_round'] == best_round
    selected = rows[data['selection']['round'] - 1]
    assert selected['full_check']['speedup'] == data['selection']['speedup']
    assert selected['accepted_commit'] == data['selection']['commit']
    assert rows[-1]['best_screen']['observed_round'] == data['selection']['fallback_terminal_round']
    assert rows[-1]['best_screen']['seconds'] > selected['full_check']['seconds']
    assert all(r['accepted_best_speedup'] < data['scope']['target_speedup'] for r in rows)


def table(data):
    lines = [BEGIN, '## Agent Optimization History', '',
             'Measured on **8×H20**, BF16 FL2VA LoRA/ZeRO-3, batch one per rank and approximately 33,750 tokens per sample. '
             'Each speed arm has two optimizer updates: update 1 is warm-up, and the interval to update 2 is timed. '
             f'**Round {data["selection"]["round"]} was the full-check best of the original Rounds 1–40 at approximately {data["selection"]["speedup"]:.2f}×, but it later failed the 50-update trajectory gate and is not the released implementation.** '
             'The released repair first passed its authoritative full check in Round 50 and was selected at the Round 60 closeout.', '',
             '![Rounds 1–40: candidate observations and cumulative full-check accepted best; Round 19 was the original-stage best, 2× target not reached](docs/performance-views/02-history.svg)', '',
             '**F** = fresh reference/candidate/reference full check (numerics, benefit and integrity). '
             '**T** = uninstrumented screening time; a timing PASS alone does not accept a candidate. '
             '**P** = profile-only observation, excluded from speed claims. **—** = no valid candidate timing. '
             'The T column selects the lowest valid candidate screening latency in each round. '
             'Accepted best changes only after F; 1.000000× in rounds 1–3 denotes the unchanged reference anchor.', '',
             '| Round | Experiment | Best T: seconds / ratio | F or other result | Accepted best at round end |',
             '|---:|---|---:|---|---:|']
    for r in data['rounds']:
        screen = r['best_screen']
        measured = f'{screen["seconds"]:.6f} / {screen["speedup"]:.6f}×' if screen else '—'
        if screen and screen['observed_round'] != r['round']:
            measured += f' (R{screen["observed_round"]})'
        if r['full_check']:
            p = r['full_check']
            result = f'**F PASS**: {p["seconds"]:.6f} s / {p["speedup"]:.6f}×. '
        elif r['classification'] == 'failure':
            result = '**Failure; no candidate time.** '
        elif r['classification'] == 'profile_only':
            p = r['profile_observation']
            result = f'**P only**: {p["seconds"]:.6f} s; no candidate time. '
        elif r['classification'] == 'no_valid_timing':
            result = '**No candidate time** (accepted-best P only). '
        else:
            result = '**T only; not accepted.** '
        result += r['outcome']
        label = f'**{r["round"]} · original-stage best**' if r['selected'] else str(r['round'])
        best = f'{r["accepted_best_speedup"]:.6f}×'
        if r['selected']:
            best = f'**{best}**'
        lines.append(f'| {label} | {r["experiment"]} | {measured} | {result} | {best} |')
    lines += ['', 'R40 includes its candidate’s terminal R41 screen; no new accepted best followed. '
              'Full-check ratios use their fresh paired-reference mean; screening ratios use the frozen reference. '
              'Profile observations and failed/missing runs never advance the accepted curve. '
              'See [data and artifact hashes](docs/performance-data/rounds-01-40.json), '
              '[denominators, cutoffs and selection audit](docs/performance-audit.md), and the '
              '[generator](docs/performance-views/generate_history.py).', '',
              'Reproduce the figure and table with `python docs/performance-views/generate_history.py`; '
              'verify checked-in outputs with `python docs/performance-views/generate_history.py --check`.', END]
    if data['scope'].get('validation_status') in ('PASS', 'FAIL'):
        lines.insert(-1, '\nOriginal Round 19 matched 50-update, three-seed trajectory gate: **' + data['scope']['validation_status'] + '**. See the [historical validation results](docs/validation-results.md). The repaired Round 50 release passed the fresh six-arm validation summarized below.')
    return '\n'.join(lines)


def svg(data):
    rows = data['rounds']
    selected = rows[data['selection']['round'] - 1]
    xmin, xmax, top, bottom = 88, 1140, 202, 540
    ymin, ymax = .70, 2.10
    x = lambda n: xmin + (n - .5) / len(rows) * (xmax - xmin)
    y = lambda v: bottom - (v - ymin) / (ymax - ymin) * (bottom - top)
    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="920" viewBox="0 0 1200 920" role="img" aria-labelledby="title desc">',
             f'<title id="title">MiniMax-H3 original optimization history: Round {data["selection"]["round"]} was the stage best at {data["selection"]["speedup"]:.6f}×</title>',
             '<desc id="desc">Forty rounds. Burgundy step curve is the cumulative full-check accepted best. Filled circles are full checks, open circles screening times, gray diamonds profile-only observations. Failures and untimed candidates have a separate status rail, not zero speedup. The 2× target was not reached. Round 40 candidate screening completed in Round 41.</desc>',
             f'<metadata>Data SHA256: {hashlib.sha256(DATA.read_bytes()).hexdigest()}</metadata>',
             '<rect width="1200" height="920" fill="white"/>',
             '<g font-family="DejaVu Sans,Arial,sans-serif" stroke-linejoin="round">']
    def text(tx, ty, value, size=15, color=INK, anchor='start', weight='normal'):
        parts.append(f'<text x="{tx:.2f}" y="{ty:.2f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{escape(str(value))}</text>')
    def line(x1,y1,x2,y2,color=GRID,width=1,extra=''):
        parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}" {extra}/>')
    def circle(tx,ty,r,fill,stroke=BURGUNDY,width=2):
        parts.append(f'<circle cx="{tx:.2f}" cy="{ty:.2f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')
    def diamond(tx,ty):
        parts.append(f'<path d="M{tx:.2f},{ty-5:.2f} L{tx+5:.2f},{ty:.2f} L{tx:.2f},{ty+5:.2f} L{tx-5:.2f},{ty:.2f} Z" fill="white" stroke="{MUTED}" stroke-width="1.6"/>')
    text(88,64,'Agent Optimization History',31,weight='bold')
    text(1140,62,f'{data["selection"]["speedup"]:.2f}×',40,BURGUNDY,'end','bold')
    text(1140,85,'original-stage full-check best',13,MUTED,'end')
    text(88,105,f'{data["scope"]["gpu_count"]} H20 GPUs · BF16 FL2VA LoRA · {data["scope"]["tokens_per_rank"]:,} tokens/rank · Rounds 1–40',17,MUTED)
    text(88,131,'Two updates per speed arm · update 1 warm-up · timed interval: completion of update 1 → update 2',14,MUTED)
    line(88,164,116,164,BURGUNDY,3);text(125,169,'Accepted best',13)
    circle(278,164,5,BURGUNDY);text(291,169,'Full check',13)
    circle(410,164,5,'white');text(423,169,'Screening',13)
    diamond(542,164);text(555,169,'Profile observation only',13)
    for a,b in [(-4,4),(4,-4)]:line(777-4,164+a,777+4,164+b,BURGUNDY,1.8)
    text(790,169,'Failure',13)
    line(890,164,900,164,MUTED,2);text(910,169,'No candidate timing',13)
    for tick in [.75,1,1.25,1.5,1.75,2]:
        line(xmin,y(tick),xmax,y(tick),GRID)
        text(xmin-12,y(tick)+5,f'{tick:.2f}×',13,MUTED,'end')
    target=data['scope']['target_speedup']
    line(xmin,y(target),xmax,y(target),BURGUNDY,1.5,'stroke-dasharray="6 6"')
    text(xmax,y(target)-10,f'{target:.1f}× target · not reached',13,BURGUNDY,'end')
    # A step curve records actual acceptance events, never the maximum of screens.
    points=[(xmin,y(rows[0]['accepted_best_speedup']))]
    previous=rows[0]['accepted_best_speedup']
    for row in rows:
        points.extend([(x(row['round']),y(previous)),(x(row['round']),y(row['accepted_best_speedup']))])
        previous=row['accepted_best_speedup']
    points.append((xmax,y(previous)))
    coords=' '.join(f'{a:.2f},{b:.2f}' for a,b in points)
    parts.append(f'<polygon points="{xmin},{bottom} {coords} {xmax},{bottom}" fill="{SOFT}" fill-opacity="0.7"/>')
    parts.append(f'<polyline points="{coords}" fill="none" stroke="{BURGUNDY}" stroke-width="3"/>')
    for row in rows:
        n=row['round'];p=row['best_screen'];f=row['full_check']
        if p:circle(x(n),y(p['speedup']),4.7,'white')
        if f:circle(x(n),y(f['speedup']),6.3,BURGUNDY)
        if not p and not f and row['profile_observation']:
            diamond(x(n),y(row['profile_observation']['speedup']))
        if not p and not f:
            if row['classification']=='failure':
                line(x(n)-4,578,x(n)+4,586,BURGUNDY,1.8)
                line(x(n)-4,586,x(n)+4,578,BURGUNDY,1.8)
            else:line(x(n)-5,582,x(n)+5,582,MUTED,2)
    for n in [1,5,10,15,19,25,30,35,40]:
        text(x(n),561,str(n),13,BURGUNDY if n==19 else MUTED,'middle','bold' if n==19 else 'normal')
    text(xmin,611,'Untimed candidate status rail: × failed gate/preflight; — profile-only or component-only round.',13,MUTED)
    sx,sy=x(selected['round']),y(selected['full_check']['speedup'])
    circle(sx,sy,12,'none','#e8baba',3);circle(sx,sy,6.3,BURGUNDY)
    line(sx+8,sy+8,675,300,BURGUNDY,1.3)
    parts.append(f'<rect x="675" y="300" width="368" height="70" rx="7" fill="white" stroke="{GRID}"/>')
    text(691,327,f'R{selected["round"]} original-stage best · full check PASS',17,BURGUNDY,weight='bold')
    text(691,352,f'{selected["full_check"]["seconds"]:.6f} s · {selected["full_check"]["speedup"]:.6f}×',16)
    text(x(40)-10,y(rows[-1]['best_screen']['speedup'])-14,'R40 → R41 screen',12,MUTED,'end')
    text(88,640,'Gray profile markers observe the current accepted version; they are not candidate speed claims.',13,MUTED)
    text(88,662,'Screens use a fixed denominator; full checks use fresh paired references. See README table and audit.',13,MUTED)
    # Compact milestone cards echo the reference view; measurements come only from data.
    cards=[(4,'Native cuDNN SDPA',f'Full check · {rows[3]["full_check"]["speedup"]:.6f}×'),
           (16,'Retention, RMSNorm and RoPE',f'Full checks through R16 · {rows[15]["full_check"]["speedup"]:.6f}×'),
           (19,'GPU / host cache headroom',f'Original-stage best · {selected["full_check"]["speedup"]:.6f}×'),
           (40,'Compiled packed SwiGLU',f'R41 screen · {rows[-1]["best_screen"]["seconds"]:.6f} s · no gain')]
    for i,(n,title,detail) in enumerate(cards):
        cx=88+(i%2)*540;cy=694+(i//2)*82
        parts.append(f'<rect x="{cx}" y="{cy}" width="512" height="70" rx="7" fill="{SOFT}"/>')
        text(cx+16,cy+29,f'R{n}',16,BURGUNDY,weight='bold')
        text(cx+76,cy+28,title,16,weight='bold')
        text(cx+76,cy+52,detail,13,MUTED)
    text(88,890,'R19 later failed the 50-update gate. A repaired implementation passed at R50 and was selected at R60.',12,MUTED)
    parts.append('</g></svg>\n')
    return '\n'.join(parts)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
    data=json.loads(DATA.read_text());validate(data)
    svg_text=svg(data);block=table(data)
    path=ROOT/'README.md';readme=path.read_text()
    if BEGIN in readme:
        start=readme.index(BEGIN);end=readme.index(END,start)+len(END)
        generated=readme[:start]+block+readme[end:]
    else:
        generated=readme.replace('## How it works',block+'\n\n## How it works',1)
    if args.check:
        assert SVG.read_text()==svg_text,'SVG is stale; regenerate'
        assert readme==generated,'README history is stale; regenerate'
        print('PASS: 40 rounds, evidence types, denominators, accepted curve, README and SVG reproduction')
    else:
        # Keep repository text files LF-only even when regenerated on Windows.
        SVG.write_text(svg_text, newline='\n');path.write_text(generated, newline='\n')


if __name__=='__main__':
    main()
