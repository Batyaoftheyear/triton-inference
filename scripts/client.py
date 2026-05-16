import argparse

import numpy as np
from PIL import Image
import tritonclient.http as httpclient


def load_image(path: str | None) -> np.ndarray:
    if path is None:
        image = np.random.rand(1, 3, 224, 224).astype(np.float32)
        return image

    image = Image.open(path).convert("RGB")
    image = image.resize((224, 224))

    image = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image = (image - mean) / std
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)

    return image.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--image", default=None)
    args = parser.parse_args()

    client = httpclient.InferenceServerClient(url=args.url)

    image = load_image(args.image)

    input_tensor = httpclient.InferInput(
        name="IMAGE",
        shape=image.shape,
        datatype="FP32",
    )
    input_tensor.set_data_from_numpy(image)

    output_tensor = httpclient.InferRequestedOutput("OUTPUT")

    response = client.infer(
        model_name="mobilenet",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )

    output = response.as_numpy("OUTPUT")

    top5 = np.argsort(output[0])[-5:][::-1]

    print("Output shape:", output.shape)
    print("Top-5 class indexes:", top5.tolist())
    print("Top-5 logits:", output[0][top5].tolist())


if __name__ == "__main__":
    main()