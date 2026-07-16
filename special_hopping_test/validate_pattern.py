"""Validate the special hopping/comb construction without running RF simulation."""

import test_settings as settings


NUM_CHANNELS = 20
NUM_BLOCKS = 10
HOPS_PER_BLOCK = 10
MIN_WRONG_ACTION_COLLISIONS = 2


def shifted_channels(hop_sequence, offset):
    return [
        (int(channel) + int(offset)) % NUM_CHANNELS
        for channel in hop_sequence
    ]


def collision_count(hop_sequence, offset, jammed_channels):
    jammed = set(jammed_channels)
    return sum(
        channel in jammed
        for channel in shifted_channels(hop_sequence, offset)
    )


def validate_pattern():
    patterns = settings.BLOCK_HOP_PATTERNS
    phase_channels = settings.COMB_PHASE_CHANNELS
    expected_offsets = settings.EXPECTED_OFFSETS

    channel_width = float(settings.ENV_CONFIG["Sub_interval"])
    comb_bandwidth = float(settings.JAMMER_CONFIG["comb"]["bandwidth"])
    comb_spacing = float(settings.JAMMER_CONFIG["comb"]["sub_interval"])
    if not (
        channel_width == comb_bandwidth
        and channel_width == comb_spacing
    ):
        raise AssertionError(
            "Channel width, comb bandwidth, and comb spacing must be equal."
        )

    if len(patterns) != NUM_BLOCKS:
        raise AssertionError(f"Expected {NUM_BLOCKS} block patterns.")
    if len(phase_channels) != 2 or len(expected_offsets) != 2:
        raise AssertionError("Exactly two comb phases are required.")

    for block_idx, pattern in enumerate(patterns):
        if len(pattern) != HOPS_PER_BLOCK:
            raise AssertionError(
                f"Block {block_idx + 1} does not contain ten hops."
            )
        if len(set(pattern)) != HOPS_PER_BLOCK:
            raise AssertionError(
                f"Block {block_idx + 1} must use ten distinct base channels."
            )
        if any(channel < 0 or channel >= NUM_CHANNELS for channel in pattern):
            raise AssertionError("Base hopping channel outside [0, 19].")

    report = []
    for phase in range(2):
        phase_report = []
        for block_idx, pattern in enumerate(patterns):
            counts = [
                collision_count(pattern, offset, phase_channels[phase])
                for offset in range(NUM_CHANNELS)
            ]
            zero_collision_offsets = [
                offset for offset, count in enumerate(counts) if count == 0
            ]
            expected = expected_offsets[phase][block_idx]
            if zero_collision_offsets != [expected]:
                raise AssertionError(
                    f"Phase {phase}, block {block_idx + 1}: expected unique "
                    f"zero-collision offset {expected}, got "
                    f"{zero_collision_offsets}."
                )

            wrong_counts = [
                count for offset, count in enumerate(counts) if offset != expected
            ]
            min_wrong = min(wrong_counts)
            if min_wrong < MIN_WRONG_ACTION_COLLISIONS:
                raise AssertionError(
                    f"Phase {phase}, block {block_idx + 1}: a wrong offset "
                    f"has only {min_wrong} collisions."
                )
            phase_report.append(min_wrong)
        report.append(phase_report)

    return report


if __name__ == "__main__":
    validation_report = validate_pattern()
    print(
        "Channel width = comb bandwidth = comb spacing = "
        f"{settings.CHANNEL_WIDTH:.0f} Hz"
    )
    for phase, minimums in enumerate(validation_report):
        print(
            f"Phase {phase}: expected offsets = "
            f"{settings.EXPECTED_OFFSETS[phase]}"
        )
        print(f"Phase {phase}: minimum wrong-action collisions = {minimums}")
    print("Special hopping pattern validation passed.")
