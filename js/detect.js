// Automatic disc-boundary detection: a REFINEMENT pass on the already-
// cropped CANVAS_SIZE x CANVAS_SIZE canvas crop.js produces, not the
// from-scratch full-photo search analysis/analyze_pores.py's
// find_disk_mask() had to do in Python. Calibration already gets the disk
// roughly centered at roughly the right size -- this exists only to correct
// for imprecise calibration (circle drawn a bit too big or too small), so
// measure.js doesn't have to blindly trust the fixed DISK_PX/2 assumption.
//
// Per the user's explicit instruction, this never silently substitutes a
// fallback boundary. If it can't find a confident match, the caller must
// refuse to grade -- see grade.js.

import { CANVAS_SIZE, DISK_PX } from './crop.js';

const HUE_MAX_DEG = 70;        // cork tan/orange vs. fixture background; equivalent to OpenCV's HUE_MAX=35 on its 0-179 scale
const MIN_DISK_VALUE = 180;    // matches analyze_pores.py's find_disk_mask -- excludes cast-shadow pixels that occasionally read low-hue by chance
const MIN_COMPACTNESS = 0.5;   // blob area / (pi * enclosingRadius^2); matches Python's MIN_DISK_COMPACTNESS
const OPEN_RADIUS = 2;         // px; small speckle-noise cleanup kernel

const EXPECTED_RADIUS = DISK_PX / 2; // 130px -- what a perfect calibration would produce
const EXPECTED_CENTER = CANVAS_SIZE / 2; // 150px
const MAX_CENTER_DRIFT = 25;         // px; how far off-center a detected disk may be before we distrust it
const MIN_RADIUS = EXPECTED_RADIUS * 0.65; // 84.5px
const MAX_RADIUS = EXPECTED_RADIUS * 1.2;  // 156px (still fits inside the 300x300 canvas)

function rgbToHsv(r, g, b) {
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const delta = max - min;
  let h = 0;
  if (delta !== 0) {
    if (max === r) h = 60 * (((g - b) / delta) % 6);
    else if (max === g) h = 60 * ((b - r) / delta + 2);
    else h = 60 * ((r - g) / delta + 4);
  }
  if (h < 0) h += 360;
  return { h, v: max };
}

function circleOffsets(radius) {
  const offsets = [];
  const r2 = radius * radius;
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      if (dx * dx + dy * dy <= r2) offsets.push([dx, dy]);
    }
  }
  return offsets;
}
const OPEN_OFFSETS = circleOffsets(OPEN_RADIUS);

/** Binary erode ('min') / dilate ('max') with the given kernel offsets. */
function binaryMorph(mask, w, h, offsets, mode) {
  const out = new Uint8Array(w * h);
  const dilate = mode === 'max';
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let hit = dilate ? 0 : 1;
      for (let k = 0; k < offsets.length; k++) {
        const nx = x + offsets[k][0];
        const ny = y + offsets[k][1];
        const v = (nx < 0 || nx >= w || ny < 0 || ny >= h) ? 0 : mask[ny * w + nx];
        if (dilate) { if (v) { hit = 1; break; } }
        else if (!v) { hit = 0; break; }
      }
      out[y * w + x] = hit;
    }
  }
  return out;
}

/** 8-connected component labeling; returns each blob's area, centroid, and max radius from that centroid. */
function componentBlobs(binaryMask, w, h) {
  const visited = new Uint8Array(w * h);
  const blobs = [];
  const stack = new Int32Array(w * h);
  for (let start = 0; start < w * h; start++) {
    if (!binaryMask[start] || visited[start]) continue;
    let sp = 0;
    stack[sp++] = start;
    visited[start] = 1;
    const pixels = [];
    while (sp > 0) {
      const cur = stack[--sp];
      const cx = cur % w;
      const cy = (cur / w) | 0;
      pixels.push([cx, cy]);
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = cx + dx, ny = cy + dy;
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
          const nIdx = ny * w + nx;
          if (binaryMask[nIdx] && !visited[nIdx]) {
            visited[nIdx] = 1;
            stack[sp++] = nIdx;
          }
        }
      }
    }
    let sx = 0, sy = 0;
    for (const [px, py] of pixels) { sx += px; sy += py; }
    const cx = sx / pixels.length;
    const cy = sy / pixels.length;
    let maxR = 0;
    for (const [px, py] of pixels) {
      const d = Math.hypot(px + 0.5 - cx, py + 0.5 - cy);
      if (d > maxR) maxR = d;
    }
    blobs.push({ area: pixels.length, cx, cy, radius: maxR });
  }
  return blobs;
}

/**
 * Find the disc's actual boundary within a normalized disk canvas (as
 * produced by crop.js's drawNormalizedDisk). Returns { cx, cy, radius,
 * detected }. `detected: false` means no confident boundary was found --
 * the caller must not grade using the fallback numbers included in that
 * case, they're for logging/debugging only.
 */
export function detectDisc(source) {
  const ctx = source.getContext ? source.getContext('2d') : null;
  const { data, width, height } = ctx
    ? ctx.getImageData(0, 0, source.width || CANVAS_SIZE, source.height || CANVAS_SIZE)
    : source;

  const candidate = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const o = i * 4;
    const { h, v } = rgbToHsv(data[o], data[o + 1], data[o + 2]);
    if (h < HUE_MAX_DEG && v > MIN_DISK_VALUE) candidate[i] = 1;
  }

  const opened = binaryMorph(binaryMorph(candidate, width, height, OPEN_OFFSETS, 'min'), width, height, OPEN_OFFSETS, 'max');
  const blobs = componentBlobs(opened, width, height);

  let best = null;
  for (const blob of blobs) {
    const enclosingArea = Math.PI * blob.radius * blob.radius;
    const compactness = enclosingArea ? blob.area / enclosingArea : 0;
    if (compactness < MIN_COMPACTNESS) continue;
    if (!best || blob.area > best.area) best = blob;
  }

  const fallback = { cx: EXPECTED_CENTER, cy: EXPECTED_CENTER, radius: EXPECTED_RADIUS, detected: false };
  if (!best) return fallback;

  const drift = Math.hypot(best.cx - EXPECTED_CENTER, best.cy - EXPECTED_CENTER);
  if (drift > MAX_CENTER_DRIFT || best.radius < MIN_RADIUS || best.radius > MAX_RADIUS) {
    return fallback;
  }

  return { cx: best.cx, cy: best.cy, radius: best.radius, detected: true };
}
