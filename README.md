# coursework

# Прогнозирование численности населения

Датасет:
https://www.kaggle.com/datasets/ahmethoso/wpp-population-by-age-and-sex

Описание:
Анализ и обработка данных по численности населения по возрастным группам.

Используемые библиотеки:
pandas, numpy, matplotlib, seaborn


# AgeVision Dataset — Распознавание возраста по изображению лица

Датасет: [UTKFace](https://susanqq.github.io/UTKFace/) + [IMDB-Wiki](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

Описание: Набор данных для обучения модели компьютерного зрения задаче классификации возраста человека по фотографии лица. Изображения разделены на 6 возрастных групп.

Используемые библиотеки: torch, torchvision, opencv-python, numpy, matplotlib, pandas

---

## 📊 Структура датасета

| Параметр | Значение |
|----------|---------|
| **Всего изображений** | 15 000 |
| **Количество классов** | 6 |
| **Формат** | JPEG, RGB |
| **Разрешение** | от 512×512 px |
| **Разделение** | Train 70% / Val 15% / Test 15% |

### Возрастные группы

| Класс | Возраст | Количество |
|-------|---------|------------|
| `age_0_2` | 0–2 года | ~2 500 |
| `age_3_12` | 3–12 лет | ~2 500 |
| `age_13_25` | 13–25 лет | ~2 500 |
| `age_26_45` | 26–45 лет | ~2 500 |
| `age_46_65` | 46–65 лет | ~2 500 |
| `age_65_plus` | 65+ лет | ~2 500 |

---

## 📁 Источники данных

| Источник | Описание | Лицензия |
|----------|----------|----------|
| [UTKFace](https://susanqq.github.io/UTKFace/) | Выровненные лица с разметкой возраста, пола и расы в имени файла | Free for academic use |
| [IMDB-Wiki](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) | Фотографии знаменитостей из фильмов и публичных мероприятий | MIT License |

---

## 🚀 Быстрый старт

### Установка зависимостей

```bash
pip install torch torchvision opencv-python numpy matplotlib pandas tqdm pyyaml
