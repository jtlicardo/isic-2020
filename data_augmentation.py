import random

import albumentations as A
import cv2
import numpy as np


def apply_geometric_transforms(image, intensity="medium"):
    h, w = image.shape[:2]
    diagonal = int(np.sqrt(h**2 + w**2))

    # Estimate "skin" color from border pixels
    top_border = image[0:5, :].reshape(-1, 3)
    bottom_border = image[-5:, :].reshape(-1, 3)
    left_border = image[:, 0:5].reshape(-1, 3)
    right_border = image[:, -5:].reshape(-1, 3)
    border_pixels = np.vstack([top_border, bottom_border, left_border, right_border])
    skin_color = np.median(border_pixels, axis=0).astype(np.uint8)

    if intensity == "light":
        rotate_limit = 30
        scale_limit = 0.1
    elif intensity == "medium":
        rotate_limit = 45
        scale_limit = 0.15
    else:
        rotate_limit = 90
        scale_limit = 0.2

    # We'll always pad to diagonal size to avoid black corners
    aug = A.Compose(
        [
            A.PadIfNeeded(
                min_height=diagonal,
                min_width=diagonal,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3 if intensity == "medium" else 0.5),
            A.Rotate(limit=rotate_limit, p=0.8, border_mode=cv2.BORDER_REFLECT_101),
            A.RandomScale(scale_limit=scale_limit, p=0.7),
            A.CenterCrop(
                height=h,
                width=w,
                p=1.0,
                pad_if_needed=True,
                border_mode=cv2.BORDER_REFLECT_101,
            ),
        ]
    )

    augmented = aug(image=image)["image"]

    # Fill any remaining black corners with the "skin_color"
    mask = np.all(augmented == [0, 0, 0], axis=-1)
    if np.any(mask):
        augmented[mask] = skin_color

    return augmented


def apply_color_transforms(image, intensity="medium"):
    """Apply color and lighting transformations based on intensity level"""
    if intensity == "light":
        aug = A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.7
                ),
                A.HueSaturationValue(
                    hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.5
                ),
                A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            ]
        )
    elif intensity == "medium":
        aug = A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.15, p=0.8
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.7
                ),
                A.GaussNoise(std_range=(0.02, 0.08), p=0.4),
                A.CLAHE(clip_limit=2.0, p=0.5),
            ]
        )
    else:
        aug = A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.9
                ),
                A.HueSaturationValue(
                    hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.8
                ),
                A.GaussNoise(std_range=(0.03, 0.1), p=0.5),
                A.CLAHE(clip_limit=4.0, p=0.6),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            ]
        )

    return aug(image=image)["image"]


def apply_elastic_transforms(image, intensity="medium"):
    """Apply elastic and affine transformations based on intensity level"""
    if intensity == "light":
        aug = A.Compose(
            [
                A.Affine(
                    scale=(0.95, 1.05),
                    translate_percent=(0.05, 0.05),
                    rotate=(-10, 10),
                    p=0.6,
                    border_mode=cv2.BORDER_REFLECT_101,
                ),
                A.ElasticTransform(alpha=0.5, sigma=25, p=0.4),
            ]
        )
    elif intensity == "medium":
        aug = A.Compose(
            [
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(0.1, 0.1),
                    rotate=(-15, 15),
                    p=0.7,
                    border_mode=cv2.BORDER_REFLECT_101,
                ),
                A.ElasticTransform(alpha=1, sigma=30, p=0.5),
                A.GridDistortion(num_steps=3, distort_limit=0.1, p=0.3),
            ]
        )
    else:
        aug = A.Compose(
            [
                A.Affine(
                    scale=(0.85, 1.15),
                    translate_percent=(0.15, 0.15),
                    rotate=(-20, 20),
                    p=0.8,
                    border_mode=cv2.BORDER_REFLECT_101,
                ),
                A.ElasticTransform(alpha=2, sigma=35, p=0.6),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.4),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.3),
            ]
        )

    return aug(image=image)["image"]


def add_synthetic_hair(image, num_hairs=20, hair_thickness=(1, 4)):
    """Add synthetic hair overlays to the image"""
    result = image.copy()
    h, w = image.shape[:2]

    r = np.random.randint(30, 70)
    g = np.random.randint(20, 50)
    b = np.random.randint(5, 30)
    hair_color = [r, g, b]

    for _ in range(num_hairs):
        # Random thickness
        thickness = random.randint(hair_thickness[0], hair_thickness[1])

        # Random starting and ending points for the hair
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(0, w), random.randint(0, h)

        # Add some curvature to make the hair look more natural
        ctrl_pt_x = (x1 + x2) // 2 + random.randint(-100, 100)
        ctrl_pt_y = (y1 + y2) // 2 + random.randint(-100, 100)

        # Create points for the curve
        t_points = np.linspace(0, 1, 100)
        points = []
        for t in t_points:
            # Quadratic Bezier curve
            x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * ctrl_pt_x + t**2 * x2
            y = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * ctrl_pt_y + t**2 * y2
            points.append((int(x), int(y)))

        prev_pt = None
        for pt in points:
            if prev_pt is not None:
                cv2.line(result, prev_pt, pt, hair_color, thickness)
            prev_pt = pt

    return result


def apply_synthetic_artifacts(image, intensity="medium"):
    """Apply synthetic artifacts like hair, blur, etc."""
    result = image.copy()

    # Add synthetic hair
    if intensity == "light":
        if random.random() < 0.3:
            result = add_synthetic_hair(result, num_hairs=random.randint(3, 8))
    elif intensity == "medium":
        if random.random() < 0.5:
            result = add_synthetic_hair(result, num_hairs=random.randint(5, 15))
    else:  # strong
        if random.random() < 0.7:
            result = add_synthetic_hair(result, num_hairs=random.randint(10, 20))

    # Apply blur
    if random.random() < 0.4:
        blur_strength = {
            "light": random.uniform(0.3, 0.6),
            "medium": random.uniform(0.5, 1.0),
            "strong": random.uniform(0.8, 1.5),
        }[intensity]
        result = cv2.GaussianBlur(result, (5, 5), blur_strength)

    # Apply compression artifacts (JPEG compression simulation)
    if random.random() < 0.3:
        quality = {
            "light": random.randint(85, 95),
            "medium": random.randint(70, 85),
            "strong": random.randint(60, 75),
        }[intensity]
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode(".jpg", result, encode_param)
        result = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

    return result


def augment_image(image, preserve_medical_features=True):
    """
    Apply a random combination of augmentations while preserving medical validity.
    Ensures that at least one augmentation is always applied.
    """
    result = image.copy()
    applied = False

    if preserve_medical_features:
        intensity_choices = ["light", "medium"]
        geo_intensity = random.choice(["light"] * 3 + ["medium"] * 2)
        color_intensity = random.choice(["light"] * 3 + ["medium"] * 2)
        elastic_intensity = random.choice(["light"] * 4 + ["medium"])
    else:
        intensity_choices = ["light", "medium", "strong"]
        geo_intensity = random.choice(intensity_choices)
        color_intensity = random.choice(intensity_choices)
        elastic_intensity = random.choice(["light", "medium"])

    if random.random() < 0.5:
        result = apply_geometric_transforms(result, intensity=geo_intensity)
        applied = True

    if random.random() < 0.7:
        result = apply_color_transforms(result, intensity=color_intensity)
        applied = True

    elastic_prob = 0.2 if preserve_medical_features else 0.3
    if random.random() < elastic_prob:
        result = apply_elastic_transforms(result, intensity=elastic_intensity)
        applied = True

    artifact_prob = 0.4 if preserve_medical_features else 0.5
    if random.random() < artifact_prob:
        artifact_intensity = random.choice(["light"] * 2 + ["medium"])
        result = apply_synthetic_artifacts(result, intensity=artifact_intensity)
        applied = True

    # Ensure at least one augmentation is applied
    if not applied:
        forced_transform = random.choice(
            [
                lambda img: apply_geometric_transforms(img, intensity=geo_intensity),
                lambda img: apply_color_transforms(img, intensity=color_intensity),
                lambda img: apply_elastic_transforms(img, intensity=elastic_intensity),
                lambda img: apply_synthetic_artifacts(
                    img, intensity=random.choice(["light", "medium"])
                ),
            ]
        )
        result = forced_transform(result)

    if result.shape[:2] != image.shape[:2]:
        result = cv2.resize(result, (image.shape[1], image.shape[0]))

    return result
