import tensorflow as tf
tf_session = tf.Session()
from keras import backend as K
K.set_session(tf_session)
from sklearn.externals import joblib
from models import create_training_model, Generator
import streamlit as st
import PIL.Image as Image
import time
# 网页配置部分
st.title('Welcome AI HaiKu!')
img = Image.open("./pic/haiku.png", "r")
st.image(img)
temperature = st.sidebar.slider('temperature:', 0.1, 0.9, 0.3, 0.1)
num = st.sidebar.slider('haiku nums:', 1, 50, 1, 1)
letter_list = [None, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
first_char = st.sidebar.selectbox('choose first char:', letter_list)
syllables = st.text_input('choose syllable', value='353', key=None)
# 提取音节格式
syllables = [int(x) for x in syllables]
run = st.button('Start')

# 检测缓存
#@st.cache(suppress_st_warning=True)
def main():
    # 加载搭建模型所需要的参数

    latent_dim, n_tokens, max_line_length, tokenizer = joblib.load('./pretrain_model/metadata.pkl')

    # 创建新的模型框架
    training_model, lstm, lines, inputs, outputs = create_training_model(latent_dim, n_tokens)

    # 将训练好的模型权重注入模型框架
    training_model.load_weights('./pretrain_model/2048-20-0.73-1.44.hdf5')

    # 利用模型创建haiku生成器
    generator = Generator(lstm, lines, tf_session, tokenizer, n_tokens, max_line_length)

    # 根据配置参数生成haiku
    for i in range(int(num)):
        st.write(generator.generate_haiku(syllables=syllables, temperature=float(temperature), first_char=first_char))

if __name__ == '__main__':

    if run:
        main()

