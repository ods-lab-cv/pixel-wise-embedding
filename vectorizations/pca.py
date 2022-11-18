from sklearn.decomposition import PCA
import numpy as np


from vectorizations.base import BaseVectorization


class PCAVectorization(BaseVectorization):
    def __init__(
            self,
            model,
            image_size,
            pca_every=10,
            device='cuda',
    ):
        super(PCAVectorization, self).__init__(
            model=model,
            image_size=image_size,
            device=device,
        )
        self.pca = PCA(n_components=3)
        self.pca_every = pca_every

    def _normalize(self, vectorization_out):
        for channel in range(3):
            cres = vectorization_out[:, :, :, channel: channel + 1]
            std = np.std(cres)
            mean = np.mean(cres)
            left = mean - 3 * std
            right = mean + 3 * std
            cres = (cres - left) / (right - left)
            cres[cres < 0] = 0
            cres[cres > 1] = 1
            vectorization_out[:, :, :, channel: channel + 1] = cres
        return vectorization_out

    def _vectorization(self, imgs, outs):
        flatten = outs.reshape(-1, outs.shape[-1])
        self.pca.fit(flatten[::self.pca_every])
        res_flatten = self.pca.transform(flatten)
        vectorization_out = res_flatten.reshape((*outs.shape[:-1], 3))
        vectorization_out = self._normalize(vectorization_out)
        return vectorization_out

