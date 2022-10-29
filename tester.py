"""
B - batch_size
C - channels = 3
(H, W) = SHAPE of img
F - FEATURE_SIZE
"""
import torch
import cv2
import numpy as np
import os
from pytorch_metric_learning import distances
from PIL import Image
from torchvision.transforms import ToTensor
from sklearn.cluster import DBSCAN
from sklearn import neighbors
from typing import Tuple, List


def gif_from_folder(folder, save_path='result.gif', duration=1000):
  files = os.listdir(folder)
  files = sorted(files, key=lambda x: int(x.split('.')[0]))
  ## Список для хранения кадров.
  frames = []

  for file in files:
    ## Открываем изображение каждого кадра.
    frame = Image.open(os.path.join(folder, file))
    ## Добавляем кадр в список с кадрами.
    frames.append(frame)
  ## Берем первый кадр и в него добавляем оставшееся кадры.
  frames[0].save(
    save_path,
    save_all=True,
    append_images=frames[1:],  ## Срез который игнорирует первый кадр.
    optimize=True,
    duration=duration,
    loop=0
  )


def get_ref(vectors_map: np.array, ref_vector: np.array) -> np.array:
  '''
  Arguments:

  vectors_map - np array (H, W, F)
  ref_vector - np array (F)

  Returns:
  dist - np array (H, W) - cosine dist between each vec in vectors_map and ref_vector
  '''
  reshape_target = vectors_map.reshape(-1, vectors_map.shape[-1])    # (H*W, F)
  dist = distances.CosineSimilarity()(torch.tensor(reshape_target), torch.tensor(ref_vector).unsqueeze(0)).detach().cpu().numpy()    # (H*W, 1)
  dist = dist.reshape((vectors_map.shape[0], vectors_map.shape[1]))
  return dist


def plot_text(image, text):
  '''
  Arguments:
  image - np array (H, W, C)
  text - str

  Returns:
  image - np array(H, W, C) - img with text
  '''
  font = cv2.FONT_HERSHEY_SIMPLEX
  org = (50, 50)
  fontScale = 1
  color = (0, 0, 255)
  thickness = 2
  image = cv2.putText(image, text, org, font, fontScale,
                      color, thickness, cv2.LINE_AA, False)
  return image


class Tester:
  def __init__(
          self,
          model,
          images_paths,
          x,
          y,
          target_b,
          save_folder,
          transforms,
          threshold=0.9,
          radius=5,
          gif_duration=500,
          run_every=100,
          device='cuda',
          plot_index=True,
  ):
    self.model = model
    self.images_paths = images_paths
    self.x = x
    self.y = y
    self.target_b = target_b
    self.transforms = transforms
    self.threshold = threshold
    self.gif_duration = gif_duration
    self.radius = radius
    self.plot_index = plot_index

    self.save_folder = save_folder
    self.device = device

    self.run_every = run_every
    self.iter = 0
    self._real_iter = 0
    self.imgs = self.read_images(self.images_paths)    # torch.FloatTensor (B, C, H, W)

  def read_images(self, images_paths: str) -> torch.FloatTensor:
    '''
    create images - torch.FloatTensor (B, C, H, W) - tensor of images from images_paths
    '''
    images = []
    for image_path in images_paths:
      im = cv2.imread(image_path)
      im = self.transforms(image=im)['image']
      im = ToTensor()(im)
      images.append(im.unsqueeze(0))
    images = torch.cat(images, dim=0)
    return images

  def predict(self, imgs: torch.FloatTensor) -> np.array:
    with torch.no_grad():
      outs = self.model(imgs.to(self.device)).detach().cpu().numpy() # np.array (B, F, H, W)
    return outs

  def plot_predicts(self, imgs: torch.FloatTensor, outs: np.ndarray) -> List[np.array]:
    '''
    Argumets: 
    imgs - torch.FloatTensor (B, C, H, W)
    outs - torch.FloatTensor (B, F, H, W)

    Returns: 
    pims - list of (im), where im - np.array (H*2, W, C) dtype = uint8
    '''
    out_np = np.moveaxis(outs, 1, -1)    # np array (B, H, W, F), dtype=float32
    x = int(self.x * imgs.shape[3]) 
    y = int(self.y * imgs.shape[2])
    pims = []
    for b, im in enumerate(imgs):    # im - (C, H, W)
      pim = (im.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8').copy()    # np array (H, W, C), dtype=uint8
      if b == self.target_b:
        pim = cv2.circle(pim.copy(), (x, y), self.radius, (255, 0, 0), 3)    # draws a circle with center at (x, y) and radius r on a target_img
      dist = get_ref(out_np[b], out_np[self.target_b, y, x])    # (H, W)
      min_val = np.min(dist)
      max_val = np.max(dist)
      dist = (dist-min_val)/(max_val-min_val)
      mask = (dist > self.threshold)    # if (dist > self.threshold)=True, then mask=1, else mask=0
      mask = mask.astype('uint8')    # mask - np array (H, W), dtype=uint8
      cntrs, _ = cv2.findContours(mask, 0, 1)
      cv2.drawContours(pim, cntrs, -1, (0, 0, 255), 3)    # draw contour on current image using a mask
      dist = (dist * 255).astype('uint8')
      dist = cv2.cvtColor(dist, cv2.COLOR_GRAY2BGR)    # (H, W, C)
      pim = cv2.cvtColor(pim, cv2.COLOR_BGR2RGB)
      pim = np.concatenate([pim, dist], axis=0)    # (H*2, W, C)
      if self.plot_index:
        pim = plot_text(pim, str(self.iter))
      pims.append(pim)
    return pims

  def save_results(self, pims: List[np.array]):
    for b in range(len(self.imgs)):
      b_name = os.path.split(self.images_paths[b].split('.')[0])[-1]
      folder_path = os.path.join(self.save_folder, b_name)
      os.makedirs(folder_path, exist_ok=True)
      cv2.imwrite(os.path.join(folder_path, str(self.iter) + '.jpg'), pims[b])
      gif_from_folder(
        folder_path,
        save_path=os.path.join(self.save_folder, b_name + '.gif'),
        duration=self.gif_duration,
      )

  def test(self):
    if self._real_iter % self.run_every == 0: # for every 500 do
      outs = self.predict(self.imgs)
      pims = self.plot_predicts(self.imgs, outs)
      self.save_results(pims)  
      self.iter += 1
    self._real_iter += 1


class DBSCANTester:
  def __init__(
          self,
          model,
          images_paths,
          save_folder,
          transforms,
          threshold=0.9,
          gif_duration=500,
          run_every=100,
          device='cuda',
          plot_index=True,
  ):
    self.model = model
    self.images_paths = images_paths
    self.transforms = transforms
    self.threshold = threshold
    self.gif_duration = gif_duration
    self.plot_index = plot_index

    self.save_folder = save_folder
    self.device = device

    self.run_every = run_every

    self.dbscan = DBSCAN(eps=120)
    self.iter = 0
    self._real_iter = 0
    self.imgs = self.read_images(self.images_paths)

  def read_images(self, images_paths):
    images = []
    for image_path in images_paths:
      im = cv2.imread(image_path)
      im = self.transforms(image=im)['image']
      im = ToTensor()(im)
      images.append(im.unsqueeze(0))
    images = torch.cat(images, dim=0)
    return images

  def predict(self, imgs):
    with torch.no_grad():
      outs = self.model(imgs.to(self.device)).detach().cpu().numpy()
    return outs

  def dbscan_find_vectors(self, out_np):
    prepare_data = out_np[::20, ::20].reshape(-1, out_np.shape[-1])
    labels = self.dbscan.fit_predict(prepare_data)
    res_vectors = []
    for u in np.unique(labels):
      mask = (labels == u)
      cur_vectors = prepare_data[mask]
      res_vectors.append(cur_vectors.mean(axis=0))
    return res_vectors

  def plot_predicts(self, imgs, outs):
    # outs = outs / np.sqrt(np.sum(np.power(outs, 2), axis=1, keepdims=True))
    out_np = np.moveaxis(outs, 1, -1)
    vectors = self.dbscan_find_vectors(out_np[0])
    color_for_vectors = [(np.random.randint(50, 200), np.random.randint(50, 200), np.random.randint(50, 200)) for v in vectors]
    pims = []
    for b in range(len(imgs)):
      pim = (imgs[b].permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8').copy()
      res_mask = np.zeros_like(pim)
      for i, v in enumerate(vectors):
        dist = get_ref(out_np[b], v)
        # dist = (dist - 0.5) * 2
        # dist[dist < 0] = 0
        mask = (dist > self.threshold).astype('uint8')
        cntrs, _ = cv2.findContours(mask, 0, 1)
        cv2.drawContours(pim, cntrs, -1, color_for_vectors[i], 3)
        cv2.drawContours(res_mask, cntrs, -1, color_for_vectors[i], -1)
      # dist = np.clip(dist, -1, 1)
      # dist = (dist + 1) / 2
      # dist = (dist * 255).astype('uint8')
      # dist = cv2.cvtColor(dist, cv2.COLOR_GRAY2BGR)
      # pim = cv2.cvtColor(pim, cv2.COLOR_BGR2RGB)
      pim = np.concatenate([pim, res_mask], axis=0)
      if self.plot_index:
        pim = plot_text(pim, str(self.iter))
      pims.append(pim)
    return pims

  def save_results(self, pims):
    for b in range(len(self.imgs)):
      b_name = os.path.split(self.images_paths[b].split('.')[0])[-1]
      folder_path = os.path.join(self.save_folder, b_name)
      os.makedirs(folder_path, exist_ok=True)
      cv2.imwrite(os.path.join(folder_path, str(self.iter) + '.jpg'), pims[b])
      gif_from_folder(
        folder_path,
        save_path=os.path.join(self.save_folder, b_name + '.gif'),
        duration=self.gif_duration,
      )

  def test(self):
    if self._real_iter % self.run_every == 0:
      outs = self.predict(self.imgs)
      pims = self.plot_predicts(self.imgs, outs)
      self.save_results(pims)
      self.iter += 1
    self._real_iter += 1
