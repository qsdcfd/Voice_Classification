{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "\bbaseline",
      "provenance": [],
      "authorship_tag": "ABX9TyM2Fx5NSa6cCZq+tlnNyrO+"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "### 필요한 라이브러리 호출\n"
      ],
      "metadata": {
        "id": "Xzl_bneIelj3"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "RT8P55dWIRO0"
      },
      "outputs": [],
      "source": [
        "#오디오 처리 라이브러리\n",
        "import librosa\n",
        "import librosa.display as dsp\n",
        "from IPython.display import Audio"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/gdrive/')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5z4JKANdISbW",
        "outputId": "b31b9cfc-6d41-4334-b199-165c9eb445ce"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/gdrive/\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#데이터 전처리\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import os\n",
        "from tqdm import tqdm"
      ],
      "metadata": {
        "id": "RbKTZT_LchTC"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#모델 재현성을 위하여 random seed고정\n",
        "import random\n",
        "\n",
        "def seed_everything(seed):\n",
        "    os.environ['PYTHONHASHSEED'] = str(seed)\n",
        "    np.random.seed(seed)\n",
        "\n",
        "seed_everything(929)"
      ],
      "metadata": {
        "id": "t1CE4r8EeNt0"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "zMKET-gJeROY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 데이터 불러오기"
      ],
      "metadata": {
        "id": "38LqQVh2eo0t"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train = pd.read_csv('/content/gdrive/MyDrive/drive-download-20220614T024615Z-001/train.csv')\n",
        "train.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "G0TWca73etek",
        "outputId": "06987569-d387-47c2-c1f2-02d7c34db35e"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "  file_name  label\n",
              "0   001.wav      9\n",
              "1   002.wav      0\n",
              "2   004.wav      1\n",
              "3   005.wav      8\n",
              "4   006.wav      0"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-f325c4ab-91db-4eec-806a-0fe91ced2c59\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>file_name</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>001.wav</td>\n",
              "      <td>9</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>002.wav</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>004.wav</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>005.wav</td>\n",
              "      <td>8</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>006.wav</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-f325c4ab-91db-4eec-806a-0fe91ced2c59')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-f325c4ab-91db-4eec-806a-0fe91ced2c59 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-f325c4ab-91db-4eec-806a-0fe91ced2c59');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "14CFwUqvetjR",
        "outputId": "3a99b556-bcfb-4761-e35c-3fcd835e3019"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 400 entries, 0 to 399\n",
            "Data columns (total 2 columns):\n",
            " #   Column     Non-Null Count  Dtype \n",
            "---  ------     --------------  ----- \n",
            " 0   file_name  400 non-null    object\n",
            " 1   label      400 non-null    int64 \n",
            "dtypes: int64(1), object(1)\n",
            "memory usage: 6.4+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "6XEjJXSXe7V0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 데이터 전처리\n",
        "\n",
        "- 소리는 보통 특정 주파수를 갖느 sin함수\n",
        "\n",
        "- 주파수 분석 기법 활용\n",
        "\n",
        "- 아날로그 데이터를 디지털 신호로 변환"
      ],
      "metadata": {
        "id": "jj0sxAdRe-__"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "data,sample_rate = librosa.load('/content/gdrive/MyDrive/Dacon/train/001.wav',sr=16000)\n",
        "print('sample_rate:', sample_rate, ',audio shape:', data.shape)\n",
        "print('length:', data.shape[0]/float(sample_rate), 'secs')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7Za6eVRHfJYv",
        "outputId": "90beaaee-2533-4063-cf9f-cc7f84d4621d"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "sample_rate: 16000 ,audio shape: (10192,)\n",
            "length: 0.637 secs\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "sampling rate의 의미는 초당 16000개(16000Hz 주파수)의 샘플을 가지고 있는 데이터라는 의미입니다. (1초에 음성 신호를 16000번 sampling)\n",
        "\n",
        "sampling rate의 defult값은 22050Hz인데, 16000Hz으로 설정한 이유는 사람의 목소리는 대부분 16000Hz 안에 포함된다고 합니다.\n",
        "\n",
        "또한 audio shape와 sampling rate를 이용해서 오디오 길이 계산을 할 수 있습니다."
      ],
      "metadata": {
        "id": "gTRGdwXRfsLr"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#train_wav데이터 프레임 생성\n",
        "def train_dataset():\n",
        "    folder = \"/content/gdrive/MyDrive/Dacon/train\"\n",
        "    dataset = []\n",
        "    for file in tqdm(os.listdir(folder),colour='green'):\n",
        "        if 'wav' in file:\n",
        "            abs_file_path = os.path.join(folder,file)\n",
        "            data, sr = librosa.load(abs_file_path, sr = 16000)\n",
        "            class_label = int(train[train.file_name == file].label)\n",
        "            dataset.append([data,class_label])\n",
        "    \n",
        "    print(\"Dataset 생성 완료\")\n",
        "    return pd.DataFrame(dataset,columns=['data','label'])"
      ],
      "metadata": {
        "id": "3Ol0Q_KwfkmT"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_wav = train_dataset()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Ua1glMjFggmc",
        "outputId": "1dd463a8-fdc6-4757-edc8-a388382f17a6"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|\u001b[32m██████████\u001b[0m| 400/400 [00:19<00:00, 20.57it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Dataset 생성 완료\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_wav.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "5lTd3kXSglSk",
        "outputId": "2dc6b259-57ff-4a7f-f2db-89dbfd60f22b"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                                data  label\n",
              "0  [3.6655838e-05, -3.7366447e-06, 3.4776433e-05,...      5\n",
              "1  [0.00011985076, 0.00016174652, 0.00017246709, ...      9\n",
              "2  [-0.00043892296, -0.00073743664, -0.0006132907...      8\n",
              "3  [-4.289015e-05, 9.891299e-05, 2.6636611e-05, 0...      4\n",
              "4  [1.2653453e-05, 2.3892262e-05, -7.51332e-06, 4...      0"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-d96f6a6d-2c7f-4eff-a54d-b251012fb26c\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>data</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>[3.6655838e-05, -3.7366447e-06, 3.4776433e-05,...</td>\n",
              "      <td>5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>[0.00011985076, 0.00016174652, 0.00017246709, ...</td>\n",
              "      <td>9</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>[-0.00043892296, -0.00073743664, -0.0006132907...</td>\n",
              "      <td>8</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>[-4.289015e-05, 9.891299e-05, 2.6636611e-05, 0...</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>[1.2653453e-05, 2.3892262e-05, -7.51332e-06, 4...</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-d96f6a6d-2c7f-4eff-a54d-b251012fb26c')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-d96f6a6d-2c7f-4eff-a54d-b251012fb26c button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-d96f6a6d-2c7f-4eff-a54d-b251012fb26c');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_wav.to_csv(\"train_wav.csv\")"
      ],
      "metadata": {
        "id": "bJOaP4Dbg8z7"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "k4mVFU_HhHPj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 음성 데이터 특징 추출\n",
        "\n",
        "- 퓨리에 변환\n",
        "\n",
        "    - 입력 신호를 다양한 주파수를 가지는 주기함수들로 분해\n",
        "\n",
        "    - 노이즈 및 배경 소리로부터 실제로 유용한 소리의 특징 추출\n",
        "\n",
        "- MFCC\n",
        "\n",
        "   - 음성데이터를 특징벡터화\n",
        "\n",
        "   - 사람이 인지하기 좋은 20~40ms로 나눠준다\n",
        "\n",
        "- Mel-scale\n",
        "\n",
        "   - 주파수 간격이 넓어지면 감지 못함\n",
        "   - filter, scaling\n",
        "\n",
        "- Argument\n",
        "\n",
        "   - y : audio data\n",
        " \n",
        "   - sr : sampling rate\n",
        "\n",
        "   - n_mfcc : return 될 mfcc의 개수를 정해주는 파라미터. 더 다양한 데이터 특징을 추출하기 위해서 증가 시킵니다.\n",
        "\n",
        "   - n_fft : frame의 length를 결정하는 파라미터. n_fft를 설정하면 window size가 디폴트 값으로 n_fft가 됩니다.\n",
        "\n",
        "   - 사람의 목소리는 대부분 16000Hz 안에 포함이 되는데, 일반적으로 자연어 처리에서는 음성을 25m의 크기를 기본으로 하고 있습니다.\n",
        "\n",
        "(ex. 16000Hz인 음성에서는 25m의 음성의 크기를 가지고 있으면 n_fft는 16000 * 0.025 = 400 (sampling rate * frame_length = n_fft)가 됩니다.)\n",
        "\n",
        "\n",
        "   - hop_length : 윈도우 길이를 나타냅니다. 길이만큼 옆으로 가면서 데이터를 읽습니다.\n",
        "\n",
        "   - hop_length도 마찬가지로 window 간의 거리이므로 sampling rate * frame_stride 가 됩니다."
      ],
      "metadata": {
        "id": "AsxxxM8ZhMTO"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def extract_features(file):\n",
        "    audio, sample_rate = librosa.load(file, sr = 16000)\n",
        "    extracted_features = librosa.feature.mfcc(y=audio,\n",
        "                                              sr=sample_rate,\n",
        "                                              n_mfcc=40)\n",
        "\n",
        "    extracted_features = np.mean(extracted_features.T,axis=0)\n",
        "    return extracted_features\n"
      ],
      "metadata": {
        "id": "iNs_VubmhziH"
      },
      "execution_count": 21,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "extract_features('/content/gdrive/MyDrive/Dacon/train/001.wav')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_qZ7d3kliUZV",
        "outputId": "2e597b69-bebe-40ac-faa2-fc6818ca755d"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([-5.4157184e+02,  1.0199717e+02, -1.0018574e+01,  4.5054619e+01,\n",
              "        7.3112831e+00,  1.0971639e+01, -1.2032939e+01, -5.8687963e+00,\n",
              "       -1.8881397e+00,  4.2930884e+00, -8.1847525e+00, -2.3072267e+00,\n",
              "       -9.1721897e+00,  1.4182716e+01, -1.2839543e+01, -3.1000307e+00,\n",
              "       -3.0502689e+00, -2.1911802e+00, -6.2639456e+00, -5.1691580e+00,\n",
              "       -1.3974123e+01,  3.3810470e+00, -6.9977813e+00,  3.7736315e-01,\n",
              "       -4.4287405e+00,  1.0799457e+00, -1.3639281e+00,  4.2418456e+00,\n",
              "        2.3687005e+00,  2.8972096e+00,  2.6670651e+00,  1.8590584e+00,\n",
              "       -3.8219376e+00, -1.6171500e-01, -1.4186366e+00, -4.1422081e+00,\n",
              "       -5.3374414e+00, -9.9907333e-01, -3.3392251e+00, -4.2987290e-01],\n",
              "      dtype=float32)"
            ]
          },
          "metadata": {},
          "execution_count": 22
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def preprocess_train_dataset(data):\n",
        "    mfccs = []\n",
        "    for i in data:\n",
        "        extracted_features = librosa.feature.mfcc(y=i,\n",
        "                                              sr=16000,\n",
        "                                              n_mfcc=40)\n",
        "        extracted_features = np.mean(extracted_features.T,axis=0)\n",
        "        mfccs.append(extracted_features)\n",
        "            \n",
        "    return mfccs\n",
        "\n",
        "mfccs = preprocess_train_dataset(train_wav.data)\n",
        "mfccs = np.array(mfccs)"
      ],
      "metadata": {
        "id": "-GvucILyijgW"
      },
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "mfccs"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "l0pbv-iTkmu4",
        "outputId": "de252cb9-ac1f-427d-a591-dba335bb3557"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[-5.0092383e+02,  1.0522231e+02, -3.0754614e+01, ...,\n",
              "         2.7968860e+00,  2.0259073e+00,  9.0582495e+00],\n",
              "       [-5.4157184e+02,  1.0199717e+02, -1.0018574e+01, ...,\n",
              "        -9.9907333e-01, -3.3392251e+00, -4.2987290e-01],\n",
              "       [-5.4121655e+02,  6.4172516e+01,  1.1352416e+01, ...,\n",
              "         9.8170495e-01, -1.2399018e+00,  1.7476790e+00],\n",
              "       ...,\n",
              "       [-6.4978589e+02,  8.1067482e+01,  2.5364252e+01, ...,\n",
              "         3.1881747e+00, -3.2863743e+00,  9.8691732e-01],\n",
              "       [-5.7801672e+02,  8.9649864e+01,  6.7791176e+00, ...,\n",
              "         3.5183573e+00, -6.8778259e-01,  2.9434044e+00],\n",
              "       [-5.1473218e+02,  6.9641624e+01, -2.2522993e+01, ...,\n",
              "        -8.1666964e-01, -7.3845682e+00, -2.4838426e+00]], dtype=float32)"
            ]
          },
          "metadata": {},
          "execution_count": 26
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "HMXQj0-Dkoy8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 변수 및 모델 정의"
      ],
      "metadata": {
        "id": "UzcM1y_wkuT3"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "train_X, test_X, train_y, test_y = train_test_split(mfccs, train_wav.label, test_size=0.4)"
      ],
      "metadata": {
        "id": "91vN-XIYkwyD"
      },
      "execution_count": 28,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print('학습시킬 train셋:', train_X.shape, train_y.shape)\n",
        "print('검증할 val 셋:', test_X.shape, test_y.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FU5oDW5VlZsP",
        "outputId": "c9d14752-c877-4785-e6e3-de67855dcd47"
      },
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "학습시킬 train셋: (240, 40) (240,)\n",
            "검증할 val 셋: (160, 40) (160,)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.ensemble import RandomForestClassifier\n",
        "\n",
        "model = RandomForestClassifier()\n",
        "\n",
        "model.fit(train_X, train_y)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LeKUwVPgllFG",
        "outputId": "e1ee84a7-5b26-498d-c225-b7fcea09685b"
      },
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "RandomForestClassifier()"
            ]
          },
          "metadata": {},
          "execution_count": 30
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "4LhP7v78l0fB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 모델 성눙"
      ],
      "metadata": {
        "id": "k_cr_X07l2ek"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "\n",
        "def ACCURACY(true, pred):\n",
        "    score= np.mean(true==pred)\n",
        "    return score"
      ],
      "metadata": {
        "id": "azgebA4Al4ed"
      },
      "execution_count": 31,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "prediction = model.predict(test_X)\n",
        "\n",
        "score = ACCURACY(test_y,  prediction)\n",
        "\n",
        "print(f'모델의 정확도는 {score*100:.2f}% 입니다')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_Qs0MxMVmBLb",
        "outputId": "5f317945-dc85-416f-9f81-112081a00274"
      },
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "모델의 정확도는 63.12% 입니다\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "1owDp93bmOcj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### test.csv분류하기"
      ],
      "metadata": {
        "id": "R3T4VAyWmXvM"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "test = pd.read_csv(\"/content/gdrive/MyDrive/drive-download-20220614T024615Z-001/test.csv\")\n",
        "test.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "Ovj9NQQsmaLs",
        "outputId": "a4f5ac1e-3413-432e-e371-d0dcb558041c"
      },
      "execution_count": 33,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "  file_name\n",
              "0   003.wav\n",
              "1   008.wav\n",
              "2   010.wav\n",
              "3   015.wav\n",
              "4   024.wav"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-76476ffb-6c29-4a2d-b376-2e80e2c4f375\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>file_name</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>003.wav</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>008.wav</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>010.wav</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>015.wav</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>024.wav</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-76476ffb-6c29-4a2d-b376-2e80e2c4f375')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-76476ffb-6c29-4a2d-b376-2e80e2c4f375 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-76476ffb-6c29-4a2d-b376-2e80e2c4f375');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 33
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def test_dataset():\n",
        "    folder = '/content/gdrive/MyDrive/test'\n",
        "    dataset=[]\n",
        "    for file in tqdm(os.listdir(folder),colour='green'):\n",
        "        if 'wav' in file:\n",
        "            abs_file_path = os.path.join(folder,file)\n",
        "            data, sr = librosa.load(abs_file_path, sr = 16000)\n",
        "            dataset.append([data, file])\n",
        "    \n",
        "    print(\"Dataset 생성 완료\")\n",
        "    return pd.DataFrame(dataset,columns=['data', 'file_name'])"
      ],
      "metadata": {
        "id": "Cls0DSugm85H"
      },
      "execution_count": 34,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test_wav = test_dataset()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hEnoeA-9nMmw",
        "outputId": "0fda5b10-33bd-4c3f-b473-cd06b5bc278b"
      },
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|\u001b[32m██████████\u001b[0m| 201/201 [00:10<00:00, 19.10it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Dataset 생성 완료\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "test_wav.to_csv('test_wav.csv')"
      ],
      "metadata": {
        "id": "pxKtLn_dnOmS"
      },
      "execution_count": 38,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "mfccs_2 = preprocess_train_dataset(test_wav.data)\n",
        "mfccs_2 = np.array(mfccs_2)"
      ],
      "metadata": {
        "id": "XSB6vmsHnSah"
      },
      "execution_count": 39,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 모델 선언\n",
        "model = RandomForestClassifier()\n",
        "\n",
        "# 모델 학습\n",
        "model.fit(mfccs, train_wav.label)\n",
        "\n",
        "# 모델 예측\n",
        "prediction = model.predict(mfccs_2)\n",
        "\n",
        "test_wav['label'] = prediction"
      ],
      "metadata": {
        "id": "oOGlll7hnhcx"
      },
      "execution_count": 40,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test_wav.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "PgkCFizdnmbK",
        "outputId": "11c56e71-d916-432f-c40f-903a3b4793af"
      },
      "execution_count": 41,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                                data file_name  label\n",
              "0  [0.00027645202, 0.00047594117, 0.0004510493, 0...   010.wav      3\n",
              "1  [0.00018057936, 0.00026723626, 0.00017172625, ...   024.wav      2\n",
              "2  [-9.729892e-05, -0.00013003226, -8.376751e-05,...   025.wav      0\n",
              "3  [-0.00012691804, -0.00019610435, -0.0001295761...   015.wav      3\n",
              "4  [0.00021518696, 0.00039842026, 0.00028533765, ...   078.wav      0"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-00efc38d-e98b-426e-8885-d131f702474b\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>data</th>\n",
              "      <th>file_name</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>[0.00027645202, 0.00047594117, 0.0004510493, 0...</td>\n",
              "      <td>010.wav</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>[0.00018057936, 0.00026723626, 0.00017172625, ...</td>\n",
              "      <td>024.wav</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>[-9.729892e-05, -0.00013003226, -8.376751e-05,...</td>\n",
              "      <td>025.wav</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>[-0.00012691804, -0.00019610435, -0.0001295761...</td>\n",
              "      <td>015.wav</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>[0.00021518696, 0.00039842026, 0.00028533765, ...</td>\n",
              "      <td>078.wav</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-00efc38d-e98b-426e-8885-d131f702474b')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-00efc38d-e98b-426e-8885-d131f702474b button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-00efc38d-e98b-426e-8885-d131f702474b');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 41
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "testset = test_wav[['file_name', 'label']]\n",
        "\n",
        "pred_df = testset.copy()\n",
        "pred_df = pred_df.sort_values(by=[pred_df.columns[0]], ascending=[True]).reset_index(drop=True)\n",
        "pred_df.head()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "SUOVgqXJnopm",
        "outputId": "91dc70d1-62a3-4fb3-b7a3-c266c1503a12"
      },
      "execution_count": 43,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "  file_name  label\n",
              "0   003.wav      0\n",
              "1   008.wav      1\n",
              "2   010.wav      3\n",
              "3   015.wav      3\n",
              "4   024.wav      2"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-cbeadf64-b034-4a20-993c-ec109a1523c0\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>file_name</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>003.wav</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>008.wav</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>010.wav</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>015.wav</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>024.wav</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-cbeadf64-b034-4a20-993c-ec109a1523c0')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-cbeadf64-b034-4a20-993c-ec109a1523c0 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-cbeadf64-b034-4a20-993c-ec109a1523c0');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 43
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "submission = pd.read_csv('/content/gdrive/MyDrive/drive-download-20220614T024615Z-001/sample_submission.csv')\n",
        "submission['label'] = pred_df['label']\n",
        "submission.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "0bMs_uj5n_dl",
        "outputId": "962ff3da-d2da-4d8b-b94a-873b2dfe69a2"
      },
      "execution_count": 44,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "  file_name  label\n",
              "0   003.wav      0\n",
              "1   008.wav      1\n",
              "2   010.wav      3\n",
              "3   015.wav      3\n",
              "4   024.wav      2"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-06a17ba0-056c-47c2-af67-167d297a7c4b\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>file_name</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>003.wav</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>008.wav</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>010.wav</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>015.wav</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>024.wav</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-06a17ba0-056c-47c2-af67-167d297a7c4b')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-06a17ba0-056c-47c2-af67-167d297a7c4b button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-06a17ba0-056c-47c2-af67-167d297a7c4b');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 44
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "submission.to_csv('submit.csv', index=False)\n"
      ],
      "metadata": {
        "id": "gZ7FEdCwoMiZ"
      },
      "execution_count": 45,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "_TUnlLKPoQQ7"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}