import keras
import argparse
import tensorflow as tf
tf_session = tf.Session()
from keras import backend as K
K.set_session(tf_session)
from sklearn.externals import joblib
from models import create_training_model, Generator

# 配置输入参数
parser = argparse.ArgumentParser(description='random generate haiku')
# 根据输入调整生成多少条haiku,默认生成1条
parser.add_argument('--num', default=1, help='henerate num')
# 根据输入生成如何音节排序的haiku.默认生成[3,5,3]音节格式haiku
parser.add_argument('--syllables', default='353', help='haiku syllable')
# 根据输入配置temperature超参数,根据经验调节出逻辑正常的haiku,默认0.3,
parser.add_argument('--temperature', default=.3, help='haiku syllable')
# 根据输入确定首字母,默认随机
parser.add_argument('--first_char', default=None, help='haiku syllable')
args = parser.parse_args()

# 加载搭建模型所需要的参数
latent_dim, n_tokens, max_line_length, tokenizer = joblib.load('./pretrain_model/metadata.pkl')
# 创建新的模型框架
training_model, lstm, lines, inputs, outputs = create_training_model(latent_dim, n_tokens)
# 将训练好的模型权重注入模型框架
training_model.load_weights('./pretrain_model/2048-20-0.73-1.44.hdf5')
# 利用模型创建haiku生成器
generator = Generator(lstm, lines, tf_session, tokenizer, n_tokens, max_line_length)



if __name__ == '__main__':


    syllables = [int(x) for x in args.syllables]
    first_char = args.first_char
    # 根据配置参数生成haiku
    for i in range(int(args.num)):
        generator.generate_haiku(syllables=syllables, temperature=float(args.temperature), first_char=first_char)
        print ()
