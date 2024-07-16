import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2gray


def create_output_folder(base_folder="C:/Users/Taha/Desktop/JPEG/output"):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(base_folder, current_time)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def apply_jpeg_compression(input_folder, output_folder, quality=40):
    # Görüntü dosyalarını oku
    image_files = os.listdir(input_folder)

    results = []

    for img_file in image_files:
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, img_file)

            # Renkli görüntüyü oku
            img_color = cv2.imread(img_path)

            if img_color is None:
                print(f"Warning: {img_path} could not be read.")
                continue

            # Renkli görüntüyü gri tonlamaya çevir
            img_gray = rgb2gray(img_color)

            # JPEG sıkıştırmasını uygula
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encimg = cv2.imencode('.jpg', img_color, encode_param)

            # Sıkıştırma sonrası görüntüyü tekrar oku (renkli)
            img_compressed_color = cv2.imdecode(encimg, 1)

            # Sıkıştırma sonrası görüntüyü gri tonlamaya çevir
            img_compressed_gray = rgb2gray(img_compressed_color)

            # SSIM hesapla (gri tonlamalı görüntüler için)
            ssim = structural_similarity(img_gray, img_compressed_gray, data_range=img_gray.max() - img_gray.min())

            # PSNR hesapla (renkli görüntüler için)
            psnr = peak_signal_noise_ratio(img_color, img_compressed_color)

            # Orijinal ve sıkıştırılmış dosyaların boyutunu al
            size_orig = os.path.getsize(img_path)
            size_comp = len(encimg)

            # Orijinal ve sıkıştırılmış dosya boyutlarını yazdır
            print(f"Original size of {img_file}: {size_orig} bytes")
            print(f"Compressed size of {img_file}: {size_comp} bytes")

            # Sıkıştırma oranı hesapla
            compression_ratio = size_orig / size_comp

            # Sıkıştırılmış görüntüyü kaydet
            compressed_img_path = os.path.join(output_folder, img_file)
            cv2.imwrite(compressed_img_path, img_compressed_color)

            # Sonuçları ekle
            results.append({
                'Image': img_file,
                'PSNR': psnr,
                'SSIM': ssim,
                'Compression Ratio': compression_ratio
            })

    return results



# Excel dosyası oluştur ve verileri yaz
def save_to_excel(results):
    # Şu anki zamanı al ve formatla
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Excel dosya adını belirle
    excel_filename = f"compression_results_{current_time}.xlsx"

    # Pandas DataFrame oluştur
    df = pd.DataFrame(results)

    # Excel dosyasına yaz
    df.to_excel(excel_filename, index=False)

    print(f"Results saved to {excel_filename}")


# Ana program
if __name__ == "__main__":
    # Örnek kullanım
    input_folder = "C:/Users/Taha/Desktop/JPEG/dataset"

    # Çıkış klasörünü oluştur
    output_folder = create_output_folder()

    # JPEG sıkıştırma ve metrik hesaplama
    results = apply_jpeg_compression(input_folder, output_folder)

    # Sonuçları ekrana yazdır
    for result in results:
        print(
            f"{result['Image']}: PSNR={result['PSNR']:.2f}, SSIM={result['SSIM']:.4f}, Compression Ratio={result['Compression Ratio']:.2f}")

    # Sonuçları Excel'e kaydet
    save_to_excel(results)
