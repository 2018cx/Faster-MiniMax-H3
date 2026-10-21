#!/usr/bin/env python3
"""Optional CPU SVG preview using installed Linux librsvg and Cairo libraries."""
import argparse
import ctypes as C
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('svg',type=Path);ap.add_argument('png',type=Path)
    args=ap.parse_args()
    rsvg=C.CDLL('librsvg-2.so.2');cairo=C.CDLL('libcairo.so.2');gobject=C.CDLL('libgobject-2.0.so.0')
    class Dimensions(C.Structure):
        _fields_=[('width',C.c_int),('height',C.c_int),('em',C.c_double),('ex',C.c_double)]
    rsvg.rsvg_handle_new_from_file.argtypes=[C.c_char_p,C.POINTER(C.c_void_p)]
    rsvg.rsvg_handle_new_from_file.restype=C.c_void_p
    rsvg.rsvg_handle_get_dimensions.argtypes=[C.c_void_p,C.POINTER(Dimensions)]
    rsvg.rsvg_handle_render_cairo.argtypes=[C.c_void_p,C.c_void_p]
    rsvg.rsvg_handle_render_cairo.restype=C.c_int
    cairo.cairo_image_surface_create.argtypes=[C.c_int,C.c_int,C.c_int]
    cairo.cairo_image_surface_create.restype=C.c_void_p
    cairo.cairo_create.argtypes=[C.c_void_p];cairo.cairo_create.restype=C.c_void_p
    cairo.cairo_surface_write_to_png.argtypes=[C.c_void_p,C.c_char_p]
    cairo.cairo_surface_write_to_png.restype=C.c_int
    cairo.cairo_destroy.argtypes=[C.c_void_p];cairo.cairo_surface_destroy.argtypes=[C.c_void_p]
    gobject.g_object_unref.argtypes=[C.c_void_p]
    error=C.c_void_p();handle=rsvg.rsvg_handle_new_from_file(str(args.svg.resolve()).encode(),C.byref(error))
    if not handle:raise RuntimeError('librsvg could not load SVG')
    dimensions=Dimensions();rsvg.rsvg_handle_get_dimensions(handle,C.byref(dimensions))
    assert 0<dimensions.width<10000 and 0<dimensions.height<10000
    surface=cairo.cairo_image_surface_create(0,dimensions.width,dimensions.height)
    context=cairo.cairo_create(surface)
    try:
        assert rsvg.rsvg_handle_render_cairo(handle,context),'librsvg render failed'
        assert cairo.cairo_surface_write_to_png(surface,str(args.png.resolve()).encode())==0
    finally:
        cairo.cairo_destroy(context);cairo.cairo_surface_destroy(surface);gobject.g_object_unref(handle)
    print(f'Rendered {dimensions.width}×{dimensions.height}: {args.png}')


if __name__=='__main__':main()
