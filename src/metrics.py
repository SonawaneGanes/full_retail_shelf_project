from collections import Counter


def generate_metrics(detections):

    brands = [d["brand"] for d in detections]

    return dict(
        Counter(brands)
    )


def estimate_shelf_share(detections):

    total_area = 0

    brand_area = {}

    for d in detections:

        x1, y1, x2, y2 = d["bbox"]

        area = (x2 - x1) * (y2 - y1)

        total_area += area

        brand_area[d["brand"]] = (
            brand_area.get(d["brand"], 0)
            + area
        )

    if total_area == 0:
        return {}

    return {
        brand: round(
            area * 100 / total_area,
            2
        )
        for brand, area
        in brand_area.items()
    }