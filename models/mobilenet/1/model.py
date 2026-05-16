import time

import numpy as np
import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.device = torch.device("cpu")
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.model.to(self.device)
        self.model.eval()

        self.processing_time_family = pb_utils.MetricFamily(
            name="custom_processing_time_seconds_total",
            description="Total processing time in seconds",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        self.processing_time_metric = self.processing_time_family.Metric(
            labels={"model": "mobilenet"}
        )

        self.requests_in_progress_family = pb_utils.MetricFamily(
            name="custom_requests_in_progress",
            description="Current number of requests in progress",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        self.requests_in_progress_metric = self.requests_in_progress_family.Metric(
            labels={"model": "mobilenet"}
        )

        self.requests_in_progress = 0

    def execute(self, requests):
        responses = []
        requests_count = len(requests)

        self.requests_in_progress += requests_count
        self.requests_in_progress_metric.set(self.requests_in_progress)
        start_time = time.perf_counter()

        try:
            for request in requests:
                try:
                    input_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE")
                    image = input_tensor.as_numpy().astype(np.float32)
                    image_tensor = torch.from_numpy(image).to(self.device)

                    with torch.no_grad():
                        output_tensor = self.model(image_tensor)

                    output_np = output_tensor.cpu().numpy().astype(np.float32)
                    response = pb_utils.InferenceResponse(
                        output_tensors=[pb_utils.Tensor("OUTPUT", output_np)]
                    )
                except Exception as exc:
                    response = pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(exc))
                    )

                responses.append(response)
        finally:
            elapsed_time = time.perf_counter() - start_time
            self.processing_time_metric.increment(elapsed_time)

            self.requests_in_progress -= requests_count
            self.requests_in_progress_metric.set(self.requests_in_progress)

        return responses

    def finalize(self):
        print("MobileNet model finalized")
