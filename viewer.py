#!/usr/bin/env python3
"""XREAL One VR180 viewer — Phase 1 (test-card + head-tracking integration).

This stage proves three things end-to-end on the Mac side, before any
video decoding is wired in:

  1. We can open a fullscreen OpenGL window on the XREAL display.
  2. The pose stream feeds head yaw/pitch/roll into the shader.
  3. Side-by-side rendering produces a coherent stereo image.

The shader draws a procedural "colored sphere" with a yaw/pitch grid
and a forward crosshair instead of sampling a video frame. When you
turn your head, the world should rotate the opposite way smoothly.

Phase 2 will replace the procedural source with a video texture
sampled through a VR180 fisheye reverse-projection.

Controls
--------
    F     toggle fullscreen
    T     zero view (set current heading as forward)
    R     recalibrate gyro
    M     cycle projection mode (testcard / fisheye / equirect)
    Up    increase rendered FOV
    Down  decrease rendered FOV
    Q     quit
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402
from OpenGL import GL as gl  # noqa: E402

from xreal_one.stream import PoseStream  # noqa: E402

VERTEX_SRC = """\
#version 330 core
layout(location=0) in vec2 aPos;
out vec2 vUv;
void main() {
    vUv = aPos * 0.5 + 0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

FRAGMENT_SRC = """\
#version 330 core
in vec2 vUv;
out vec4 fragColor;

uniform float uViewportWidth;
uniform float uHeadYaw;       // degrees
uniform float uHeadPitch;
uniform float uHeadRoll;
uniform float uTanHalfFovX;
uniform float uTanHalfFovY;
uniform int   uProjMode;       // 0 testcard, 1 fisheye, 2 equirect
uniform int   uShowDebug;

const float PI = 3.14159265359;

mat3 rotY(float a) { float c=cos(a), s=sin(a); return mat3(c,0,-s, 0,1,0, s,0,c); }
mat3 rotX(float a) { float c=cos(a), s=sin(a); return mat3(1,0,0, 0,c,-s, 0,s,c); }
mat3 rotZ(float a) { float c=cos(a), s=sin(a); return mat3(c,-s,0, s,c,0, 0,0,1); }

vec3 testCardColor(vec3 d) {
    // d is the world-space view direction (after head rotation).
    // Convert to "world" yaw/pitch.
    float yaw   = degrees(atan(d.x, d.z));
    float pitch = degrees(asin(clamp(d.y, -1.0, 1.0)));

    // Base color from direction so motion is obvious.
    vec3 base = 0.5 + 0.5 * d;

    // Grid lines every 15 degrees.
    float yaw_dist   = abs(mod(yaw   + 7.5, 15.0) - 7.5);
    float pitch_dist = abs(mod(pitch + 7.5, 15.0) - 7.5);
    float grid = smoothstep(0.6, 0.0, min(yaw_dist, pitch_dist));
    base = mix(base, vec3(1.0), grid * 0.6);

    // Major axes (yaw=0, pitch=0) bold red.
    if (abs(yaw) < 0.5 || abs(pitch) < 0.5) base = mix(base, vec3(1.0, 0.2, 0.2), 0.85);

    // Forward crosshair (within 4 deg of +Z).
    if (abs(yaw) < 4.0 && abs(pitch) < 4.0) {
        if (abs(yaw) < 0.4 || abs(pitch) < 0.4) base = vec3(1.0, 1.0, 0.2);
    }

    return base;
}

void main() {
    bool isLeftEye = gl_FragCoord.x < uViewportWidth * 0.5;
    float halfX = isLeftEye ? gl_FragCoord.x : (gl_FragCoord.x - uViewportWidth * 0.5);
    vec2 eyeUv = vec2(halfX / (uViewportWidth * 0.5), vUv.y);

    // Build a ray in camera space from per-eye NDC.
    vec2 ndc = eyeUv * 2.0 - 1.0;
    vec3 dir = normalize(vec3(ndc.x * uTanHalfFovX, ndc.y * uTanHalfFovY, 1.0));

    // Apply head rotation: we rotate the world the OPPOSITE way of the head,
    // so when the user turns their head right, the scene appears to slide left.
    float yaw   = radians(uHeadYaw);
    float pitch = radians(uHeadPitch);
    float roll  = radians(uHeadRoll);
    mat3 R = rotY(yaw) * rotX(pitch) * rotZ(roll);
    vec3 d = R * dir;

    vec3 col;
    if (uProjMode == 0) {
        // Phase 1: head-tracked test card, no video needed.
        col = testCardColor(d);
    } else if (uProjMode == 1) {
        // Phase 2 placeholder: VR180 equiangular fisheye reverse-projection.
        // Stubbed to test card until video texture is wired up.
        col = testCardColor(d);
    } else {
        // Phase 2 placeholder: VR180 equirectangular reverse-projection.
        col = testCardColor(d);
    }

    // Edge-of-eye separator for visual confirmation of the SBS split.
    if (uShowDebug == 1) {
        if (abs(gl_FragCoord.x - uViewportWidth * 0.5) < 1.5) col = vec3(0.0, 1.0, 0.0);
    }

    fragColor = vec4(col, 1.0);
}
"""


def _compile_shader(src: str, kind: int) -> int:
    sid = gl.glCreateShader(kind)
    gl.glShaderSource(sid, src)
    gl.glCompileShader(sid)
    if not gl.glGetShaderiv(sid, gl.GL_COMPILE_STATUS):
        log = gl.glGetShaderInfoLog(sid).decode("utf-8", "replace")
        raise RuntimeError(f"shader compile failed: {log}")
    return sid


def _link_program(vert_src: str, frag_src: str) -> int:
    prog = gl.glCreateProgram()
    vs = _compile_shader(vert_src, gl.GL_VERTEX_SHADER)
    fs = _compile_shader(frag_src, gl.GL_FRAGMENT_SHADER)
    gl.glAttachShader(prog, vs)
    gl.glAttachShader(prog, fs)
    gl.glLinkProgram(prog)
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
        log = gl.glGetProgramInfoLog(prog).decode("utf-8", "replace")
        raise RuntimeError(f"program link failed: {log}")
    return prog


def _make_fullscreen_quad() -> Tuple[int, int]:
    verts = np.array(
        [-1, -1,  1, -1,  -1, 1,  -1, 1,  1, -1,  1, 1],
        dtype=np.float32,
    )
    vao = gl.glGenVertexArrays(1)
    vbo = gl.glGenBuffers(1)
    gl.glBindVertexArray(vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STATIC_DRAW)
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    gl.glBindVertexArray(0)
    return vao, vbo


def _pick_display(want_index: Optional[int]) -> Tuple[int, Tuple[int, int]]:
    sizes = pygame.display.get_desktop_sizes()
    print("detected displays:")
    for i, (w, h) in enumerate(sizes):
        print(f"  [{i}] {w}x{h}")

    if want_index is not None:
        if want_index < 0 or want_index >= len(sizes):
            raise SystemExit(f"--display {want_index} out of range (0..{len(sizes) - 1})")
        return want_index, sizes[want_index]

    for i, (w, h) in enumerate(sizes):
        if w >= 3000:  # XREAL Full SBS (3840x1080 typical)
            print(f"auto-picked display [{i}] (likely XREAL Full SBS)")
            return i, (w, h)
    print("no wide display found; using primary [0]")
    return 0, sizes[0]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--display", type=int, default=None,
                   help="display index (default: auto-pick a >=3000 wide one)")
    p.add_argument("--windowed", action="store_true",
                   help="run in a 1920x540 window instead of fullscreen")
    p.add_argument("--no-tracker", action="store_true",
                   help="don't connect to glasses; render with zero pose (for offline GL test)")
    p.add_argument("--fov", type=float, default=50.0,
                   help="initial vertical field of view in degrees (default 50)")
    args = p.parse_args()

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
    )
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_FLAGS, pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG
    )
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

    display_idx, native_size = _pick_display(args.display)
    if args.windowed:
        size = (1920, 540)
        flags = pygame.OPENGL | pygame.DOUBLEBUF
    else:
        size = native_size
        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.NOFRAME
    try:
        screen = pygame.display.set_mode(size, flags, display=display_idx, vsync=1)
    except TypeError:
        # Older pygame without `display=` kwarg.
        os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
        screen = pygame.display.set_mode(size, flags, vsync=1)
    pygame.display.set_caption("xreal-one viewer (Phase 1: test card)")

    print(f"GL context {gl.glGetString(gl.GL_VERSION).decode()}")
    print(f"GLSL       {gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION).decode()}")

    program = _link_program(VERTEX_SRC, FRAGMENT_SRC)
    vao, _vbo = _make_fullscreen_quad()
    u_viewport_width = gl.glGetUniformLocation(program, "uViewportWidth")
    u_yaw   = gl.glGetUniformLocation(program, "uHeadYaw")
    u_pitch = gl.glGetUniformLocation(program, "uHeadPitch")
    u_roll  = gl.glGetUniformLocation(program, "uHeadRoll")
    u_fovx  = gl.glGetUniformLocation(program, "uTanHalfFovX")
    u_fovy  = gl.glGetUniformLocation(program, "uTanHalfFovY")
    u_mode  = gl.glGetUniformLocation(program, "uProjMode")
    u_debug = gl.glGetUniformLocation(program, "uShowDebug")

    # Tracker
    pose_stream: Optional[PoseStream] = None
    if not args.no_tracker:
        pose_stream = PoseStream()
        pose_stream.start()

    fov_y_deg = args.fov
    proj_mode = 0
    show_debug = True
    is_fullscreen = not args.windowed

    clock = pygame.time.Clock()
    last_status_print = 0.0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_t and pose_stream is not None:
                    pose_stream.zero_view()
                    print("\n[zeroed]")
                elif event.key == pygame.K_r and pose_stream is not None:
                    pose_stream.recalibrate()
                    print("\n[recalibrating]")
                elif event.key == pygame.K_m:
                    proj_mode = (proj_mode + 1) % 3
                    name = ["testcard", "fisheye(stub)", "equirect(stub)"][proj_mode]
                    print(f"\n[proj mode = {proj_mode} {name}]")
                elif event.key == pygame.K_d:
                    show_debug = not show_debug
                elif event.key == pygame.K_UP:
                    fov_y_deg = min(120.0, fov_y_deg + 2.0)
                    print(f"\n[fov_y = {fov_y_deg:.1f}]")
                elif event.key == pygame.K_DOWN:
                    fov_y_deg = max(20.0, fov_y_deg - 2.0)
                    print(f"\n[fov_y = {fov_y_deg:.1f}]")
                elif event.key == pygame.K_f:
                    is_fullscreen = not is_fullscreen
                    new_flags = pygame.OPENGL | pygame.DOUBLEBUF | (
                        pygame.NOFRAME if is_fullscreen else 0
                    )
                    new_size = native_size if is_fullscreen else (1920, 540)
                    try:
                        screen = pygame.display.set_mode(
                            new_size, new_flags, display=display_idx, vsync=1
                        )
                    except TypeError:
                        screen = pygame.display.set_mode(new_size, new_flags, vsync=1)

        w, h = pygame.display.get_window_size()
        gl.glViewport(0, 0, w, h)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        gl.glUseProgram(program)
        # Per-eye horizontal FOV: each eye fills half the window. With native
        # 1:1 per-eye aspect (e.g. 960x960 source) and the SBS framebuffer
        # being 2:1 (3840x1080), the per-eye target aspect is (w/2)/h.
        eye_aspect = (w * 0.5) / max(1, h)
        tan_half_fov_y = float(np.tan(np.radians(fov_y_deg) * 0.5))
        tan_half_fov_x = tan_half_fov_y * eye_aspect

        if pose_stream is not None:
            snap = pose_stream.latest()
            rel = snap.relative
            yaw, pitch, roll = rel.yaw_deg, rel.pitch_deg, rel.roll_deg
            calib = snap.calibration_progress
            calibrated = snap.is_calibrated
            connected = snap.connected
        else:
            yaw = pitch = roll = 0.0
            calib = 1.0
            calibrated = True
            connected = False

        gl.glUniform1f(u_viewport_width, float(w))
        gl.glUniform1f(u_yaw, yaw)
        gl.glUniform1f(u_pitch, pitch)
        gl.glUniform1f(u_roll, roll)
        gl.glUniform1f(u_fovx, tan_half_fov_x)
        gl.glUniform1f(u_fovy, tan_half_fov_y)
        gl.glUniform1i(u_mode, proj_mode)
        gl.glUniform1i(u_debug, 1 if show_debug else 0)

        gl.glBindVertexArray(vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

        pygame.display.flip()

        now = time.monotonic()
        if now - last_status_print > 0.5:
            if pose_stream is not None and not calibrated:
                pct = int(calib * 100)
                bar = "#" * (pct // 5) + "." * (20 - pct // 5)
                sys.stdout.write(f"\rcalibrating [{bar}] {pct:3d}%   ")
            else:
                conn = "connected" if connected else "disconnected"
                sys.stdout.write(
                    f"\r{conn}  yaw {yaw:+7.2f}  pitch {pitch:+7.2f}  roll {roll:+7.2f}  "
                    f"fov {fov_y_deg:4.1f}  fps {clock.get_fps():4.1f}   "
                )
            sys.stdout.flush()
            last_status_print = now

        clock.tick(120)

    print()
    if pose_stream is not None:
        pose_stream.stop()
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
