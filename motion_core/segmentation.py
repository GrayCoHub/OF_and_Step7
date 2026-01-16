# segmentation.py

from typing import List, Tuple


from typing import List, Tuple, Optional


def _segment_span_len(seg: Tuple[int, int]) -> int:
    return int(seg[1] - seg[0] + 1)


def _count_kept_in_span(kept: set, start: int, end: int) -> int:
    s = int(start)
    e = int(end)
    return sum(1 for k in kept if s <= int(k) <= e)


def _segment_density(kept: set, seg: Tuple[int, int]) -> Tuple[int, int, float]:
    span_len = _segment_span_len(seg)
    md_count = _count_kept_in_span(kept, seg[0], seg[1])
    density = (md_count / span_len) if span_len > 0 else 0.0
    return int(md_count), int(span_len), float(density)


def _longest_by_span(segments: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if not segments:
        return None
    return max(segments, key=lambda se: (_segment_span_len(se), -int(se[0])))


def _segments_from_kept_frames(kept_frames: List[int], config: "Config") -> List[Tuple[int, int]]:
    if not kept_frames:
        return []
    kept_frames = sorted(int(x) for x in kept_frames)
    segments: List[Tuple[int, int]] = []
    start = prev = kept_frames[0]
    for f in kept_frames[1:]:
        if f <= prev + int(config.MERGE_GAP_FRAMES):
            prev = f
            continue
        if (prev - start + 1) >= int(config.MIN_SEGMENT_LENGTH):
            segments.append((start, prev))
        start = prev = f
    if (prev - start + 1) >= int(config.MIN_SEGMENT_LENGTH):
        segments.append((start, prev))
    return segments
