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
uniform int   uProjMode;       // 0 testcard, 1 fisheye, 2 equirect, 3 flat-sbs, 4 flat-tb
uniform int   uShowDebug;
uniform int   uHasVideo;
uniform sampler2D uVideo;
uniform float uFisheyeFovDeg;  // physical FOV of the fisheye lens (typ 180)
uniform int   uFlipY;          // 0 = sample with image-space v, 1 = flip
uniform int   uRot90;          // 0/1/2/3 = 0/90/180/270 degree CCW per-eye rotation
uniform float uCalibProgress;  // -1 = hide overlay, 0..1 = calibration progress

const float PI = 3.14159265359;

mat3 rotY(float a) { float c=cos(a), s=sin(a); return mat3(c,0,-s, 0,1,0, s,0,c); }
mat3 rotX(float a) { float c=cos(a), s=sin(a); return mat3(1,0,0, 0,c,-s, 0,s,c); }
mat3 rotZ(float a) { float c=cos(a), s=sin(a); return mat3(c,-s,0, s,c,0, 0,0,1); }

// Returns true if `d` (unit vector) is within `radDeg` of `axis`.
bool nearAxis(vec3 d, vec3 axis, float radDeg) {
    return acos(clamp(dot(d, axis), -1.0, 1.0)) < radians(radDeg);
}

vec3 testCardColor(vec3 d) {
    float yaw   = degrees(atan(d.x, d.z));
    float pitch = degrees(asin(clamp(d.y, -1.0, 1.0)));

    // Soft direction-coded background so motion is always obvious.
    vec3 base = 0.4 + 0.4 * d;

    // 15-deg yaw/pitch grid (only on the forward hemisphere; the back is darker).
    float yaw_dist   = abs(mod(yaw   + 7.5, 15.0) - 7.5);
    float pitch_dist = abs(mod(pitch + 7.5, 15.0) - 7.5);
    float grid = smoothstep(0.6, 0.0, min(yaw_dist, pitch_dist));
    base = mix(base, vec3(1.0), grid * 0.4);
    if (d.z < 0.0) base *= 0.5;  // back hemisphere dimmer

    // Six fixed-color cardinal markers (10-deg radius each).
    //   world +Y up      RED
    //   world -Y down    BLUE
    //   world -X left    GREEN
    //   world +X right   YELLOW
    //   world +Z forward WHITE
    //   world -Z behind  MAGENTA (so we can spot when something flipped 180)
    if (nearAxis(d, vec3( 0,  1,  0), 10.0)) base = vec3(1.0, 0.1, 0.1);
    if (nearAxis(d, vec3( 0, -1,  0), 10.0)) base = vec3(0.1, 0.3, 1.0);
    if (nearAxis(d, vec3(-1,  0,  0), 10.0)) base = vec3(0.1, 1.0, 0.3);
    if (nearAxis(d, vec3( 1,  0,  0), 10.0)) base = vec3(1.0, 1.0, 0.2);
    if (nearAxis(d, vec3( 0,  0,  1),  6.0)) base = vec3(1.0);
    if (nearAxis(d, vec3( 0,  0, -1),  6.0)) base = vec3(1.0, 0.2, 1.0);

    // Forward crosshair (within 4 deg of +Z, thin lines).
    if (abs(yaw) < 4.0 && abs(pitch) < 4.0) {
        if (abs(yaw) < 0.4 || abs(pitch) < 0.4) base = vec3(1.0, 1.0, 0.2);
    }
    return base;
}

// Rotate the lookup direction around the +Z (forward) axis. This corrects
// for sources whose lens frame ("up" in image) is rotated relative to world
// up — typical in re-encoded uploads. Doing the rotation in direction space
// keeps yaw/pitch/roll mapped to their true world axes; doing it in UV
// space (the previous approach) would swap the perceived yaw and pitch.
vec3 rotateAroundForward(vec3 d, int rot) {
    if (rot == 1) return vec3(-d.y,  d.x, d.z);  // 90 CCW
    if (rot == 2) return vec3(-d.x, -d.y, d.z);  // 180
    if (rot == 3) return vec3( d.y, -d.x, d.z);  // 270 CCW = 90 CW
    return d;
}

// Sample SBS video texture at per-eye UV in [0,1]^2 (image-space, top-left
// origin: y=0 means top of source frame). Source layout: left-eye in left
// half (u in [0,0.5]), right-eye in right half.
vec3 sampleSbs(vec2 uvEye, bool isLeftEye) {
    if (uvEye.x < 0.0 || uvEye.x > 1.0 || uvEye.y < 0.0 || uvEye.y > 1.0) {
        return vec3(0.0);
    }
    float uOffset = isLeftEye ? 0.0 : 0.5;
    float v = (uFlipY == 1) ? (1.0 - uvEye.y) : uvEye.y;
    return texture(uVideo, vec2(uvEye.x * 0.5 + uOffset, v)).rgb;
}

// Top-bottom (over/under) SBS sampling: top half is left eye, bottom right.
vec3 sampleFlatTb(vec2 uvEye, bool isLeftEye) {
    if (uvEye.x < 0.0 || uvEye.x > 1.0 || uvEye.y < 0.0 || uvEye.y > 1.0) {
        return vec3(0.0);
    }
    float v = isLeftEye ? (uvEye.y * 0.5) : (uvEye.y * 0.5 + 0.5);
    if (uFlipY == 1) v = 1.0 - v;
    return texture(uVideo, vec2(uvEye.x, v)).rgb;
}

// VR180 equiangular fisheye reverse-projection.
// Source: a circular fisheye that maps a `uFisheyeFovDeg` cone of directions
// to an inscribed circle. Equiangular means r in image is proportional to
// theta (angle from the lens forward axis).
//
// Returns the per-eye image-space UV, plus an `outside` flag.
vec2 fisheyeUv(vec3 d, out bool outside) {
    outside = false;
    // theta from forward axis (+Z): 0 in front, pi at directly behind.
    float theta = acos(clamp(d.z, -1.0, 1.0));
    float halfFov = radians(uFisheyeFovDeg) * 0.5;
    if (theta > halfFov) { outside = true; return vec2(0.5); }
    float phi = atan(d.y, d.x);
    float r = theta / halfFov;          // 0 center, 1 fisheye edge
    // Image-space v: y=0 is top of source frame. World-up (positive d.y)
    // gives positive sin(phi); we want it to land near v=0 (top), so subtract.
    return vec2(0.5 + 0.5 * r * cos(phi), 0.5 - 0.5 * r * sin(phi));
}

// VR180 equirectangular reverse-projection (180 horizontal x 180 vertical).
// Source maps (yaw in [-90,90], pitch in [-90,90]) to a square per eye.
vec2 equirectUv(vec3 d, out bool outside) {
    outside = false;
    float phi   = atan(d.x, d.z);              // [-pi, pi], forward = 0
    float theta = asin(clamp(d.y, -1.0, 1.0)); // [-pi/2, pi/2]
    if (abs(phi) > PI * 0.5) { outside = true; return vec2(0.5); }
    return vec2(0.5 + phi / PI, 0.5 - theta / PI);
}

void main() {
    bool isLeftEye = gl_FragCoord.x < uViewportWidth * 0.5;
    float halfX = isLeftEye ? gl_FragCoord.x : (gl_FragCoord.x - uViewportWidth * 0.5);
    vec2 eyeUv = vec2(halfX / (uViewportWidth * 0.5), vUv.y);

    vec2 ndc = eyeUv * 2.0 - 1.0;
    vec3 dir = normalize(vec3(ndc.x * uTanHalfFovX, ndc.y * uTanHalfFovY, 1.0));

    float yaw   = radians(uHeadYaw);
    float pitch = radians(uHeadPitch);
    float roll  = radians(uHeadRoll);
    mat3 R = rotY(yaw) * rotX(pitch) * rotZ(roll);
    vec3 d = R * dir;

    // Bring the lookup direction into the source frame's lens orientation.
    // Test card uses raw d (world frame); video sampling uses lens-frame d.
    vec3 dLens = rotateAroundForward(d, uRot90);

    vec3 col;
    if (uProjMode == 0 || uHasVideo == 0) {
        col = testCardColor(d);
    } else if (uProjMode == 1) {
        bool outside;
        vec2 uvImg = fisheyeUv(dLens, outside);
        col = outside ? vec3(0.05) : sampleSbs(uvImg, isLeftEye);
    } else if (uProjMode == 2) {
        bool outside;
        vec2 uvImg = equirectUv(dLens, outside);
        col = outside ? vec3(0.05) : sampleSbs(uvImg, isLeftEye);
    } else if (uProjMode == 3) {
        // Flat side-by-side: source left half = left eye, right half = right
        // eye. Head tracking is intentionally ignored - the screen stays
        // glued to the viewport, which feels like watching a fixed 3D screen.
        // eyeUv is in screen space (y=0 at screen bottom); sampleSbs expects
        // image space (y=0 at frame top), so flip v on the way in.
        col = sampleSbs(vec2(eyeUv.x, 1.0 - eyeUv.y), isLeftEye);
    } else {
        // Flat top-bottom (over/under). Same screen->image v flip.
        col = sampleFlatTb(vec2(eyeUv.x, 1.0 - eyeUv.y), isLeftEye);
    }

    if (uShowDebug == 1) {
        if (abs(gl_FragCoord.x - uViewportWidth * 0.5) < 1.5) col = vec3(0.0, 1.0, 0.0);
    }

    // Calibration overlay: a centered horizontal progress bar drawn in
    // per-eye screen space (eyeUv). Visible in both eyes so the user can't
    // miss it when wearing the glasses. Hidden after calibration finishes.
    if (uCalibProgress >= 0.0) {
        float bx0 = 0.30, bx1 = 0.70;
        float by0 = 0.46, by1 = 0.49;
        if (eyeUv.x > bx0 && eyeUv.x < bx1 && eyeUv.y > by0 && eyeUv.y < by1) {
            float fill = bx0 + clamp(uCalibProgress, 0.0, 1.0) * (bx1 - bx0);
            col = (eyeUv.x < fill) ? vec3(0.2, 1.0, 0.4) : vec3(0.15, 0.15, 0.18);
        }
        // Thin inner border so the bar is readable on any background.
        bool nearTop    = abs(eyeUv.y - by1) < 0.0015 && eyeUv.x >= bx0 && eyeUv.x <= bx1;
        bool nearBottom = abs(eyeUv.y - by0) < 0.0015 && eyeUv.x >= bx0 && eyeUv.x <= bx1;
        bool nearLeft   = abs(eyeUv.x - bx0) < 0.0008 && eyeUv.y >= by0 && eyeUv.y <= by1;
        bool nearRight  = abs(eyeUv.x - bx1) < 0.0008 && eyeUv.y >= by0 && eyeUv.y <= by1;
        if (nearTop || nearBottom || nearLeft || nearRight) col = vec3(1.0);
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
    p.add_argument("video", nargs="?", default=None,
                   help="path to a VR180 SBS video file (omitted = test card only)")
    p.add_argument("--display", type=int, default=None,
                   help="display index (default: auto-pick a >=3000 wide one)")
    p.add_argument("--windowed", action="store_true",
                   help="run in a 1920x540 window instead of fullscreen")
    p.add_argument("--no-tracker", action="store_true",
                   help="don't connect to glasses; render with zero pose")
    p.add_argument("--no-mag", action="store_true",
                   help="disable magnetometer-based yaw correction "
                        "(useful if your environment has strong magnetic interference)")
    p.add_argument("--fov", type=float, default=87.0,
                   help="initial diagonal field of view in degrees (default 87). "
                        "57 = XREAL One Pro 1:1 angular mapping (subjects feel "
                        "smaller/farther); 87 = roughly the previous wide-sweep "
                        "default. Lower = more zoom, higher = wider.")
    p.add_argument("--proj", choices=["testcard", "fisheye", "equirect", "flat-sbs", "flat-tb"],
                   default=None,
                   help="initial projection mode (default: fisheye if video given, else testcard). "
                        "fisheye/equirect = VR180 spherical sources; flat-sbs/flat-tb = 3D "
                        "rectangular videos (no head tracking applied to the image)")
    p.add_argument("--decode-width", type=int, default=None,
                   help="downscale decoded frames to this width (default: source width)")
    p.add_argument("--decode-height", type=int, default=None,
                   help="downscale decoded frames to this height (default: source height)")
    p.add_argument("--fisheye-fov", type=float, default=180.0,
                   help="physical FOV of the fisheye lens in degrees (default 180)")
    p.add_argument("--mute", action="store_true",
                   help="don't play audio (default plays via ffplay/afplay)")
    p.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                   help="rotate per-eye sampling by N degrees CCW (default 0)")
    p.add_argument("--flip-y", action="store_true",
                   help="vertically flip the per-eye sampling")
    p.add_argument("--invert-pitch", action="store_true", default=True,
                   help="invert pitch sign (default true; XREAL One pitch axis is "
                        "opposite the rotX-around-+X convention used in the shader)")
    p.add_argument("--no-invert-pitch", dest="invert_pitch", action="store_false")
    p.add_argument("--invert-yaw", action="store_true",
                   help="invert yaw sign")
    p.add_argument("--invert-roll", action="store_true", default=True,
                   help="invert roll sign (default true; matches the pitch convention)")
    p.add_argument("--no-invert-roll", dest="invert_roll", action="store_false")
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
        # FULLSCREEN takes the display exclusively so the dock and menu bar
        # don't overlap; size matches native_size so SDL doesn't trigger a
        # mode change.
        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN
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
    u_yaw    = gl.glGetUniformLocation(program, "uHeadYaw")
    u_pitch  = gl.glGetUniformLocation(program, "uHeadPitch")
    u_roll   = gl.glGetUniformLocation(program, "uHeadRoll")
    u_fovx   = gl.glGetUniformLocation(program, "uTanHalfFovX")
    u_fovy   = gl.glGetUniformLocation(program, "uTanHalfFovY")
    u_mode   = gl.glGetUniformLocation(program, "uProjMode")
    u_debug  = gl.glGetUniformLocation(program, "uShowDebug")
    u_hasvid = gl.glGetUniformLocation(program, "uHasVideo")
    u_video  = gl.glGetUniformLocation(program, "uVideo")
    u_fisheye_fov = gl.glGetUniformLocation(program, "uFisheyeFovDeg")
    u_flip_y = gl.glGetUniformLocation(program, "uFlipY")
    u_rot90  = gl.glGetUniformLocation(program, "uRot90")
    u_calib  = gl.glGetUniformLocation(program, "uCalibProgress")

    # Video setup
    video_stream = None
    video_tex = 0
    video_tex_size: Tuple[int, int] = (0, 0)
    last_uploaded_pts = -1.0
    if args.video is not None:
        from xreal_one.video import VideoStream  # lazy: avoids importing av in --no-video runs
        video_stream = VideoStream(
            args.video,
            target_width=args.decode_width,
            target_height=args.decode_height,
            loop=True,
        )
        video_stream.start()
        video_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, video_tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

    # Audio (external process, no PyAV decode for audio path)
    audio_player = None
    if args.video is not None and not args.mute:
        from xreal_one.audio import AudioPlayer
        audio_player = AudioPlayer(args.video, loop=True)
        audio_player.start()
        if audio_player.backend:
            print(f"audio backend: {audio_player.backend}")

    # Tracker
    pose_stream: Optional[PoseStream] = None
    if not args.no_tracker:
        pose_stream = PoseStream(use_mag=not args.no_mag)
        pose_stream.start()

    fov_diag_deg = args.fov
    if args.proj is not None:
        proj_mode = {
            "testcard": 0, "fisheye": 1, "equirect": 2,
            "flat-sbs": 3, "flat-tb": 4,
        }[args.proj]
    else:
        proj_mode = 1 if args.video is not None else 0
    show_debug = True
    flip_y = args.flip_y
    rot90 = (args.rotate // 90) % 4
    invert_pitch = args.invert_pitch
    invert_yaw = args.invert_yaw
    invert_roll = args.invert_roll
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
                    proj_mode = (proj_mode + 1) % 5
                    name = ["testcard", "fisheye", "equirect",
                            "flat-sbs", "flat-tb"][proj_mode]
                    print(f"\n[proj mode = {proj_mode} {name}]")
                elif event.key == pygame.K_d:
                    show_debug = not show_debug
                elif event.key == pygame.K_v:
                    flip_y = not flip_y
                    print(f"\n[flip_y = {flip_y}]")
                elif event.key == pygame.K_o:
                    rot90 = (rot90 + 1) % 4
                    print(f"\n[rotate = {rot90 * 90} deg]")
                elif event.key == pygame.K_p:
                    invert_pitch = not invert_pitch
                    print(f"\n[invert_pitch = {invert_pitch}]")
                elif event.key == pygame.K_y:
                    invert_yaw = not invert_yaw
                    print(f"\n[invert_yaw = {invert_yaw}]")
                elif event.key == pygame.K_l:
                    invert_roll = not invert_roll
                    print(f"\n[invert_roll = {invert_roll}]")
                elif event.key == pygame.K_UP:
                    fov_diag_deg = max(20.0, fov_diag_deg - 2.0)
                    print(f"\n[fov_diag = {fov_diag_deg:.1f}]")
                elif event.key == pygame.K_DOWN:
                    fov_diag_deg = min(120.0, fov_diag_deg + 2.0)
                    print(f"\n[fov_diag = {fov_diag_deg:.1f}]")
                elif event.key == pygame.K_0:
                    fov_diag_deg = args.fov
                    print(f"\n[fov_diag reset to {fov_diag_deg:.1f}]")
                elif event.key == pygame.K_f:
                    is_fullscreen = not is_fullscreen
                    new_flags = pygame.OPENGL | pygame.DOUBLEBUF | (
                        pygame.FULLSCREEN if is_fullscreen else 0
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

        # Pull and (re)upload latest video frame.
        if video_stream is not None:
            latest = video_stream.latest()
            if latest is not None:
                arr, pts = latest
                if pts != last_uploaded_pts:
                    last_uploaded_pts = pts
                    fh, fw = arr.shape[0], arr.shape[1]
                    gl.glBindTexture(gl.GL_TEXTURE_2D, video_tex)
                    if (fw, fh) != video_tex_size:
                        gl.glTexImage2D(
                            gl.GL_TEXTURE_2D, 0, gl.GL_RGB, fw, fh, 0,
                            gl.GL_RGB, gl.GL_UNSIGNED_BYTE, arr,
                        )
                        video_tex_size = (fw, fh)
                    else:
                        gl.glTexSubImage2D(
                            gl.GL_TEXTURE_2D, 0, 0, 0, fw, fh,
                            gl.GL_RGB, gl.GL_UNSIGNED_BYTE, arr,
                        )

        gl.glUseProgram(program)
        # `fov_diag_deg` is the user-facing diagonal FOV. Convert to per-eye
        # vertical/horizontal half-FOV-tangents using the per-eye aspect:
        #   tan(diag/2) = sqrt(aspect^2 + 1) * tan(vfov/2)
        # so tan(vfov/2) = tan(diag/2) / sqrt(aspect^2 + 1).
        eye_aspect = (w * 0.5) / max(1, h)
        tan_half_diag = float(np.tan(np.radians(fov_diag_deg) * 0.5))
        tan_half_fov_y = tan_half_diag / float(np.sqrt(eye_aspect * eye_aspect + 1.0))
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

        if invert_yaw:   yaw = -yaw
        if invert_pitch: pitch = -pitch
        if invert_roll:  roll = -roll

        gl.glUniform1f(u_viewport_width, float(w))
        gl.glUniform1f(u_yaw, yaw)
        gl.glUniform1f(u_pitch, pitch)
        gl.glUniform1f(u_roll, roll)
        gl.glUniform1f(u_fovx, tan_half_fov_x)
        gl.glUniform1f(u_fovy, tan_half_fov_y)
        gl.glUniform1i(u_mode, proj_mode)
        gl.glUniform1i(u_debug, 1 if show_debug else 0)
        gl.glUniform1i(u_hasvid, 1 if video_stream is not None and video_tex_size[0] > 0 else 0)
        gl.glUniform1f(u_fisheye_fov, args.fisheye_fov)
        gl.glUniform1i(u_flip_y, 1 if flip_y else 0)
        gl.glUniform1i(u_rot90, rot90)
        # Show overlay only while not yet calibrated; -1 hides it.
        gl.glUniform1f(u_calib, calib if pose_stream is not None and not calibrated else -1.0)
        if video_stream is not None:
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, video_tex)
            gl.glUniform1i(u_video, 0)

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
                conn = "tr-conn" if connected else "tr-down"
                vid_part = ""
                if video_stream is not None:
                    fc = video_stream.frame_count
                    sw, sh = video_stream.source_size
                    vid_part = f"  vid {sw}x{sh}->{video_tex_size[0]}x{video_tex_size[1]} fr#{fc}"
                proj_name = ["test", "fish", "equi", "fSBS", "fTB"][proj_mode]
                sys.stdout.write(
                    f"\r{conn}  yaw{yaw:+6.1f} pit{pitch:+6.1f} rol{roll:+6.1f}"
                    f"  fov{fov_diag_deg:4.1f} mode={proj_name}{vid_part}  fps{clock.get_fps():4.1f} "
                )
            sys.stdout.flush()
            last_status_print = now

        clock.tick(120)

    print()
    if audio_player is not None:
        audio_player.stop()
    if video_stream is not None:
        video_stream.stop()
    if pose_stream is not None:
        pose_stream.stop()
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
