import numpy as np
from pytorch_metric_learning import distances
import torch
import typing as tp


from vectorizations.base import BaseVectorization


class InteractiveVectorization(BaseVectorization):
    def __init__(
            self,
            model: torch.nn.Module,
            image_size: tp.Tuple[int, int],
            device: str = 'cuda',
    ):
        super(InteractiveVectorization, self).__init__(
            model=model,
            image_size=image_size,
            device=device,
        )

        self.x = None
        self.y = None
        self.batch_i = None

    def set_pos(self, x: tp.Union[int, float], y: tp.Union[int, float], batch_i: tp.Union[int, float]):
        self.x = int(x)
        self.y = int(y)
        self.batch_i = int(batch_i)

    def _out_to_distance(self, batch_out: np.ndarray, ref_vector: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _vectorization(self, imgs: np.ndarray, outs: np.ndarray) -> np.ndarray:
        results = []
        for b in range(len(outs)):
            distance = self._out_to_distance(outs[b], outs[self.batch_i, self.y, self.x])
            results.append(distance)
        return np.stack(results, axis=0)


class InteractiveVectorizationCosine(InteractiveVectorization):
    def _out_to_distance(self, batch_out: np.ndarray, ref_vector: np.ndarray) -> np.ndarray:
        if (self.x is None) or (self.y is None) or (self.batch_i is None):
            raise AttributeError(f'x={self.x} y={self.y} batch_i={self.batch_i}')
        reshape_target = batch_out.reshape(-1, batch_out.shape[-1])
        dist = distances.CosineSimilarity()(torch.tensor(reshape_target),
                                            torch.tensor(ref_vector).unsqueeze(0)).detach().cpu().numpy()
        dist = dist.reshape((batch_out.shape[0], batch_out.shape[1]))
        return dist

