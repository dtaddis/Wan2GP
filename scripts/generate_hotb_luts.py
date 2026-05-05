from __future__ import annotations

import colorsys
from pathlib import Path


SIZE = 33


def hue_dist(hue: float, center: float) -> float:
    return abs((hue - center + 0.5) % 1.0 - 0.5) * 360.0


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge0 == edge1:
        return 1.0 if value >= edge1 else 0.0
    t = max(0.0, min(1.0, (value - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def mask_near(hue: float, centers: list[float], radius: float, feather: float) -> float:
    distance = min(hue_dist(hue, center) for center in centers)
    return 1.0 - smoothstep(radius, radius + feather, distance)


def make_lut(
    path: Path,
    *,
    skin_sat: float,
    world_sat: float,
    cool_sat: float,
    green_sat: float,
    red_hue_push: float,
) -> None:
    with path.open("w", encoding="ascii", newline="\n") as lut:
        lut.write(f'TITLE "{path.stem}"\n')
        lut.write(f"LUT_3D_SIZE {SIZE}\n")
        lut.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        lut.write("DOMAIN_MAX 1.0 1.0 1.0\n")
        for b_i in range(SIZE):
            b = b_i / (SIZE - 1)
            for g_i in range(SIZE):
                g = g_i / (SIZE - 1)
                for r_i in range(SIZE):
                    r = r_i / (SIZE - 1)
                    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
                    skin = (
                        mask_near(hue, [0.015, 0.055, 0.985], 24.0, 28.0)
                        * smoothstep(0.05, 0.22, saturation)
                        * smoothstep(0.08, 0.22, lightness)
                        * (1.0 - smoothstep(0.86, 1.0, lightness))
                    )
                    green = mask_near(hue, [0.24, 0.32, 0.39], 34.0, 38.0)
                    cool = mask_near(hue, [0.52, 0.60, 0.67], 40.0, 45.0)
                    red = mask_near(hue, [0.0, 0.985], 20.0, 28.0)
                    sat_factor = world_sat
                    sat_factor += (green_sat - world_sat) * green
                    sat_factor += (cool_sat - world_sat) * cool
                    sat_factor = sat_factor * (1.0 - skin) + skin_sat * skin
                    saturation_out = max(0.0, min(1.0, saturation * sat_factor))
                    hue_out = (hue + red_hue_push * skin * red) % 1.0
                    r_out, g_out, b_out = colorsys.hls_to_rgb(hue_out, lightness, saturation_out)
                    lut.write(f"{r_out:.6f} {g_out:.6f} {b_out:.6f}\n")


def main() -> None:
    out_dir = Path("outputs") / "luts"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_lut(
        out_dir / "HotB_SkinSafe_Natural_33.cube",
        skin_sat=0.72,
        world_sat=1.10,
        cool_sat=1.13,
        green_sat=1.13,
        red_hue_push=0.010,
    )
    make_lut(
        out_dir / "HotB_SkinSafe_Stronger_33.cube",
        skin_sat=0.58,
        world_sat=1.14,
        cool_sat=1.18,
        green_sat=1.17,
        red_hue_push=0.016,
    )
    print(out_dir.resolve())


if __name__ == "__main__":
    main()
