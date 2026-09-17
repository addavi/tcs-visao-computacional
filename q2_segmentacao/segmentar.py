import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters, morphology, measure, color, segmentation
from scipy import ndimage as ndi
import cv2


imagem = data.coins()  
# Suavizar levemente para reduzir ruído antes da limiarização
imagem_suave = filters.gaussian(imagem, sigma=1)

# Limiarização automática com o método de Otsu (transforma imagem em preto e branco)
limiar = filters.threshold_otsu(imagem_suave)
binaria = imagem_suave > limiar

# remover ruído pequeno e fechar buracos internos
binaria_limpa = morphology.remove_small_objects(binaria, min_size=100)
binaria_limpa = morphology.remove_small_holes(binaria_limpa, area_threshold=100)

# Distância euclidiana( medida do segmento de reta mais curto entre dois pontos para achar os "centros" das moedas via Watershed)
distancia = ndi.distance_transform_edt(binaria_limpa)

# Encontrar máximos locais = marcadores para o Watershed
coordenadas_maximos = morphology.local_max = None
from skimage.feature import peak_local_max
coords = peak_local_max(distancia, min_distance=20, labels=binaria_limpa)
mascara_marcadores = np.zeros(distancia.shape, dtype=bool)
mascara_marcadores[tuple(coords.T)] = True
marcadores, _ = ndi.label(mascara_marcadores)

# separa moedas que estão encostadas
rotulos = segmentation.watershed(-distancia, marcadores, mask=binaria_limpa)

# Contagem de objetos (moedas)
num_moedas = len(np.unique(rotulos)) - 1  # -1 porque o rótulo 0 é o fundo
print(f"Total de moedas detectadas: {num_moedas}")


fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(imagem, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(binaria_limpa, cmap='gray')
ax[1].set_title('Segmentação binária (Otsu)')
imagem_colorida = color.label2rgb(rotulos, image=imagem, bg_label=0)
ax[2].imshow(imagem_colorida)
ax[2].set_title(f'Watershed — {num_moedas} moedas')
for a in ax:
    a.axis('off')

plt.tight_layout()
plt.savefig('saida/moedas_segmentadas.png', dpi=150)
plt.show()


imagem_bgr = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
contornos = measure.find_contours(binaria_limpa, 0.5)
for contorno in contornos:
    contorno = np.fliplr(contorno).astype(np.int32)
    cv2.polylines(imagem_bgr, [contorno], isClosed=True, color=(0, 255, 0), thickness=2)
cv2.imwrite('saida/moedas_contornos.png', imagem_bgr)