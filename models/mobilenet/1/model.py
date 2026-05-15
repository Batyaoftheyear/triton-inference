import time

import numpy as np
import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.device = torch.device("cpu")

        weights = MobileNet_V2_Weights.DEFAULT
        self.model = mobilenet_v2(weights=weights)
        self.model.to(self.device)
        self.model.eval()

        # Метрика суммарного времени обработки запросов
        self.time_metric_family = pb_utils.MetricFamily(
            name="custom_processing_time_seconds_total",
            description="Total processing time in seconds",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        self.time_metric = self.time_metric_family.Metric(
            labels={"model": "mobilenet"}
        )

        # Метрика текущего числа запросов в обработке
        self.requests_metric_family = pb_utils.MetricFamily(
            name="custom_requests_in_progress",
            description="Current number of requests in progress",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        self.requests_metric = self.requests_metric_family.Metric(
            labels={"model": "mobilenet"}
        )

        self.requests_in_progress = 0

    def execute(self, requests):
        responses = []

        self.requests_in_progress += len(requests)
        self.requests_metric.set(self.requests_in_progress)

        start_time = time.perf_counter()

        try:
            for request in requests:
                input_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE")
                image = input_tensor.as_numpy().astype(np.float32)

                image = torch.from_numpy(image).to(self.device)

                with torch.no_grad():
                    output = self.model(image)

                output = output.cpu().numpy().astype(np.float32)
                output_tensor = pb_utils.Tensor("OUTPUT", output)

                responses.append(
                    pb_utils.InferenceResponse(output_tensors=[output_tensor])
                )

        finally:
            elapsed_time = time.perf_counter() - start_time

            self.time_metric.increment(elapsed_time)

            self.requests_in_progress -= len(requests)
            self.requests_metric.set(self.requests_in_progress)

        return responses

    def finalize(self):
        print("MobileNet model finalized")