"""Line-aware sliding-window chunker with overlap."""
from typing import List


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> List[str]:
    # break very long lines so nothing exceeds `size`
    lines: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        while len(line) > size:
            lines.append(line[:size])
            line = line[size:]
        if line:
            lines.append(line)

    chunks: List[str] = []
    current: List[str] = []
    length = 0
    for line in lines:
        if current and length + len(line) + 1 > size:
            chunks.append("\n".join(current))
            carry: List[str] = []          # trailing lines kept as overlap
            c_len = 0
            for prev in reversed(current):
                if c_len + len(prev) + 1 > overlap:
                    break
                carry.insert(0, prev)
                c_len += len(prev) + 1
            current, length = carry, c_len
        current.append(line)
        length += len(line) + 1
    if current:
        last = "\n".join(current)
        if not chunks or last != chunks[-1]:
            chunks.append(last)
    return chunks
