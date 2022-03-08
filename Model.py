import modeling as modeling
import optimization
from utils import *
import Chuncker
import DataHolder

def gelu(x):
    #활성화함수. relu와 비슷한데 relu는 음수는 아예 죽여버리는데 이건 조금은 살려둠
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def seq_length(sequence):
    #RNN 쓸때 필요 / sequence 길이 
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)

def masked_softmax(logits, mask, dim):
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30)  # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask)  # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist

def get_variables_with_name(name, train_only=True, printable=False):
    """Get variable list by a given name scope.
    Examples
    ---------
    dense_vars = tl.layers.get_variable_with_name('dense', True, True)
    """
    print("  [*] geting variables with %s" % name)
    # tvar = tf.trainable_variables() if train_only else tf.all_variables()
    if train_only:
        t_vars = tf.trainable_variables()
    else:
        try: # TF1.0
            t_vars = tf.global_variables()
        except: # TF0.12
            t_vars = tf.all_variables()

    d_vars = [var for var in t_vars if name in var.name]
    if printable:
        for idx, v in enumerate(d_vars):
            print("  got {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))
    return d_vars


class KoNET:
    def __init__(self, firstTraining, testCase=False):
        #chuncker는 뭐하는 건지
        #chunckr text overlap 점수로 만들어주는. overlap 많을수록 1에 가깝게 출력됨. 
        self.chuncker = Chuncker.Chuncker()
        self.first_training = firstTraining

        self.save_path = 'C:\\Users\\goldjunyeong\\Desktop\\textdetection\\latest_save\\new\\model.ckpt'
        self.bert_path = 'C:\\Users\\goldjunyeong\\Desktop\\textdetection\\roberta_base\\roberta_base.ckpt'

        self.input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_segments = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_mask = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.dialog_label = tf.placeholder(shape=[None, None], dtype=tf.float32)

        self.processor = DataHolder.DataHolder()
        #keep_prob는 뭔지
        #DataHolder은 단순히 그냥 만들어진 ids, segment파일 관리?
        #drop : 많이 잘릴수록(0에 가까울수록) 시간 오래걸리는데. test 잘 나옴
        self.keep_prob = 0.9
        if testCase is True:
            self.keep_prob = 1.0

        self.testCase = testCase

        self.sess = None
        self.prediction_start = None
        self.prediction_stop = None

        self.column_size = 15
        self.row_size = 50
        #여기서 사용하는, korean_conversation 처리할 때 electratokenizer 사용?
        #그럼 makeanswr에서 komoran 사용하는데 이건 별 문제 없을지? 다 통일해야 된는건가? 아 근데 makeanswer은 그냥 자체적으로 지들끼리 비교하고 similarity 계산하는거니까 괜찮겠당
        from transformers import ElectraTokenizer
        self.tokenizer_ = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

        #drop은 뭔지 #여기서는 안쓰임
        self.drop1 = 0.2
        self.drop2 = 0.15
        self.drop3 = 0.1
        self.drop4 = 0.05
        #testCase는 뭔지 model.test돌릴 때만 true?
        if testCase is True:
            self.drop1 = 0
            self.drop2 = 0
            self.drop3 = 0
            self.drop4 = 0

    def model_setting(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        #session. gpu에 데이터 올려서. tensor을 연산하기 위한. 변수 .
        self.sess = tf.Session(config=config)
        #bert_config는 bert 모델 정보? 같은거 같은데 config는 무엇인지 /gpu
        bert_config = modeling.BertConfig.from_json_file('bert_config_mecab_base.json')
        #input_mask는 어디에 쓰는 건지 >길이 남을 때 padding 처리
        input_mask = tf.where(self.input_ids > 0, tf.ones_like(self.input_ids),
                              tf.zeros_like(self.input_ids))

        model = modeling.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=self.input_ids,
            input_mask=input_mask,
            token_type_ids=self.input_segments,
            scope='roberta',
        )
        #이건 두 문장 사이 그 probability [3.1 , 28.7] 이런거 구하는거 ?? ㅇㅇ
        _, _, probs = self.get_next_sentence_output(bert_config, model.get_pooled_output(), self.dialog_label)
        #get next sentence output 에서 변수 3개 나오는데 앞에 2개는 안써서 안받는거.
        self.probs = probs
        self.sess.run(tf.initialize_all_variables())

        saver = tf.train.Saver()
        saver.restore(self.sess, self.save_path)

    def make_propagate(self, sentence1, sentence2):
        input_ids = np.zeros(shape=[1, 128], dtype=np.int32)
        input_segments = np.zeros(shape=[1, 128], dtype=np.int32)

        tokens = ['[CLS]']
        segments = ['0']

        tokens_1 = self.tokenizer_.tokenize(sentence1)
        #sentence1 토크나이즈해서 각 토큰마다 돌아가며
        for token in tokens_1:
            #tokens에 저장
            #segment는 다 0인지?
            tokens.append(token)
            segments.append(0)
        #sentence1에 있는 토큰 다 저장한 후에 [SEP] 저장 하고 다시 sentence2도 같은 절차
        tokens.append('[SEP]')
        segments.append(0)

        tokens_2 = self.tokenizer_.tokenize(sentence2)
        for token in tokens_2:
            tokens.append(token)
            segments.append(1)
        tokens.append('[SEP]')
        segments.append(1)
        
        ids = self.tokenizer_.convert_tokens_to_ids(tokens=tokens)
        #convert_tokens_to_ids 는 각 토큰을 원래 있는 단어 사전에서 해당하는 단어의 위치로 ? 바꾸는건지?
        #예를 들어 컴퓨터->345 / 안녕->197 이런 식으로?
        length = len(ids)
        if length > 128:
            length = 128
        #length 128개가 최대. 그럼 만약에 length가 128이 안될 경우에는 패딩 처리

        for i in range(length):
            #앞서 input_ids= np.zeros(shape[1,128]) 만들어놓은거에 토크나이즈한 인덱스값? 넣기
            #그럼 여기서 input_segments는 다 1인감 length 128 안될 경우에 빈칸은 0으로 남고?
            input_ids[0, i] = ids[i]

            ######여기 바꿔쩡 위가 상현씨
            input_segments[0, i] = segments[i]
            #input_segments[0, i] = 1

        #feed_dict은 뭔가요 place holder에 어떤 데이터 넣을지. 명시
        feed_dict = {self.input_ids: input_ids,
                     self.input_segments: input_segments}

        probs = self.sess.run(self.probs, feed_dict=feed_dict)

        return probs

    def get_qa_loss(self, logit1):
        """Get loss and log probs for the next sentence prediction."""

        # Simple binary classification. Note that 0 is "next sentence" and 1 is
        # "random sentence". This weight matrix is not used after pre-training.
        with tf.variable_scope("cls/seq_relationship"):
            loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit1, labels=self.dialog_label)
        return loss1
    #cls/seq_relationship이 뭔지..? 이건 그 training돌렸을 때 나오는 loss값인지

    #여기선 안쓰임
    def get_qa_probs2(self, model_output, is_training=False):
        """Get loss and log probs for the next sentence prediction."""
        #make_propagate랑 다른지? 들어가는 input이 달라서 다른 것 같긴 한뎅
        keep_prob = 0.9
        #keep_prob는 뭔지
        #training안할 때(test나 propagate)는 왜 1로 하고 training할때는 0.9로 두는지
        if is_training is False:
            keep_prob = 1.0

        #MRC_block이 뭐하는 건지. 여기는 뭐하는 건지.
        #model_output은 뭘 뜻하는지 / fully_connected는 layer을 만드는 거,,? 몬가 hidden layer느낌//? 그럼 layer 3개?
        with tf.variable_scope("MRC_block"):
            model_output = Fully_Connected(model_output, output=768, name='hidden1', activation=gelu)
            model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

            model_output = Fully_Connected(model_output, output=512, name='hidden2', activation=gelu)
            model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

            model_output = Fully_Connected(model_output, output=256, name='hidden3', activation=gelu)
            model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

        with tf.variable_scope("pointer_net1"):
            log_probs_s = Fully_Connected(model_output, output=2, name='prediction', activation=None, reuse=False)
            #output 2개가 그 흐름 안끊김 / 끊김에 대한 probability인지?

        return log_probs_s

    def get_next_sentence_output(self, bert_config, input_tensor, labels):
        """Get loss and log probs for the next sentence prediction."""

        # Simple binary classification. Note that 0 is "next sentence" and 1 is
        # "random sentence". This weight matrix is not used after pre-training.

        #with tf.variable_scope("output"):
        with tf.variable_scope("cls/predictions"):
            with tf.variable_scope("prediction_layer"):
                hidden_states_ = input_tensor
                #prediction이 최종 layer인지? 그럼 앞서 log_probs_s랑은 다른건지? 이 결과로 나오는게 #[3.7112, 21.9990] 이런건지?
                prediction = Fully_Connected(hidden_states_, output=2, name='prediction_layer',
                                             activation=None,
                                             reuse=False)
            #[3.7112, 21.9990] => [0.1< 0.9]  target: [0, 1]

            #앞선 log_probs_s랑 뭐가 다른지? (softmax는 합 1로 만드는거)
            log_probs = tf.nn.log_softmax(prediction, axis=-1)
            #labels = tf.reshape(labels, [-1])
            #one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
            #cross_entropy loss 값 구하는 함수 reduce_sum
            per_example_loss = -tf.reduce_sum(labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            #총 평균 loss 계산 reduce_mean
            return loss, per_example_loss, prediction

    def Training(self, is_Continue, training_epoch):
        #
        dropout = 0.2
        #
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.99

        bert_config = modeling.BertConfig.from_json_file('bert_config_mecab_base.json')

        with tf.Session(config=config) as sess:
            input_mask = tf.where(self.input_ids > 0, tf.ones_like(self.input_ids),
                                  tf.zeros_like(self.input_ids))
            #input_mask의 의미 /
            model = modeling.BertModel(
                config=bert_config,
                is_training=True,               #dropout =0.2 자동으로
                input_ids=self.input_ids,
                input_mask=input_mask,
                token_type_ids=self.input_segments,
                scope='roberta',
            )

            bert_variables = tf.global_variables()
            #roberta pretrain 된거 사용하기 위해

            loss, _, _ = self.get_next_sentence_output(bert_config, model.get_pooled_output(), self.dialog_label)
            total_loss = tf.reduce_mean(loss)
            #그 training시킬때 프린트되는 값이 total_loss

            learning_rate = 2e-5
            #여기서 learning rate를 바꾸는게 결과에 의미가 있을지?
            optimizer = optimization.create_optimizer(loss=total_loss, init_lr=learning_rate, num_train_steps=25000,
                                                      num_warmup_steps=500, use_tpu=False)
            #loss 감소시키는 방향으로 weight 업데이트
            sess.run(tf.initialize_all_variables())

            if self.first_training is True:
                #pretrain 된거 불러올 때 
                saver = tf.train.Saver(bert_variables)
                saver.restore(sess, self.bert_path)
                print('BERT restored')

            if is_Continue is True:
                saver = tf.train.Saver()
                saver.restore(sess, self.save_path)
                #이어서 학습

            for i in range(training_epoch):
                ##
                # 라벨 정보 바뀌어야함
                ##
                input_ids, input_segments, dialog_label = \
                    self.processor.next_batch()

                feed_dict = {self.input_ids: input_ids,
                             self.input_segments: input_segments,
                             self.dialog_label: dialog_label
                             }

                loss_, _ = sess.run([total_loss, optimizer], feed_dict=feed_dict)
                print(i, loss_)
                if (i % 2000 == 0 or i == training_epoch-1):
                    print('saved!')
                    saver = tf.train.Saver()
                    saver.save(sess, self.save_path)

    def Test(self, is_Continue, training_epoch):
        cor = 0

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.99

        bert_config = modeling.BertConfig.from_json_file('bert_config_mecab_base.json')

        with tf.Session(config=config) as sess:
            input_mask = tf.where(self.input_ids > 0, tf.ones_like(self.input_ids),
                                  tf.zeros_like(self.input_ids))

            model = modeling.BertModel(
                config=bert_config,
                is_training=False,
                input_ids=self.input_ids,
                input_mask=input_mask,
                token_type_ids=self.input_segments,
                scope='roberta',
            )

            _, _, probs = self.get_next_sentence_output(bert_config, model.get_pooled_output(), self.dialog_label)
            probs = tf.nn.softmax(probs, axis=1)
            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            saver.restore(sess, self.save_path)

            print('BERT restored')

            file = open('prob_check', 'w', encoding='utf-8')
            for i in range(training_epoch):
                ##
                # 라벨 정보 바뀌어야함
                ##
                input_ids, input_segments, dialog_label = \
                    self.processor.test_batch()

                feed_dict = {self.input_ids: input_ids,
                             self.input_segments: input_segments,
                             self.dialog_label: dialog_label
                             }

                prediction_probs = sess.run(probs, feed_dict=feed_dict)

                #이거 의미?ㅜㅜ
                if (prediction_probs[0, 0] > prediction_probs[0, 1]) or (prediction_probs[1, 0] < prediction_probs[1, 1]):
                    cor += 1

                #ROC curve 만들기 위한 용도
                file.write(str(prediction_probs[0, 0]) + '\t' + '1' + '\n')
                file.write(str(prediction_probs[1, 0]) + '\t' + '0' + '\n')
                #왜 (i+1)*2로 cor값을 나눠주는지. 
                print(i, ':', cor / ((i+ 1) * 2))
                print(prediction_probs)

