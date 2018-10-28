# 构造简单的LSTM模型

由于人们在思考时不会从头开始，而是保留之前思考的一些结果为现在的决策提供支持。循环神经网络正是利用了这一特性，保留前一个神经元的状态来训练。RNN最大 的特点就是神经元的某些输出可作为其输入再次传输到神经元中，因此可以利用之前的信息。

虽然RNN被设计成为可以处理整个时间序列信息，但其记忆最深的还是最后输入的一些信号。而更早之前的信号的强度则越来越低，最后只能起到一点辅助的作用，即决定RNN输出的还是最后的一些信号。为此LSTM模型被提议出来来解决这个问题。

下面我们建立一个简单的LSTM模型

首先我们导入一些库：

```python
import time
import numpy as np
import tensorflow as tf
import reader
```

然后我们来定义输入的类：

```python
 __input__()相当于类的构造方法
#num_steps是LSTM的展开步数 然后计算每个epoch的大小
class PTBInput(object):
    def __init__(self,config,data,name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps  #返回一个不大于该数的整数 因为遍历是从0开始到 epoch-1
        self.input_data, self.targets = reader.ptb_producer(
            data,batch_size,num_steps, name=name)
```

定义模型的类：

```python
class PTBModel(object):
    def __init__(self,is_training,config,input_):
        self._input = input_
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size #节点数
        vocab_size = config.vocab_size#词汇表大小
    #设置默认的LSTM单元
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
             size,forget_bias=0.0,state_is_tuple=True)

        attn_cell = lstm_cell#attn_cell引用lstm_cell 如果if不成了就直接是lstm_cell

        #使用MultiRNNCell将前面构造的lstm_cell多层堆叠得到cell
        #堆叠次数为congif中的num_layers
        if is_training and config.keep_prob < 1:#如果在训练状态且keep_prob小于1 则在lstm_cell之后接一个Dropout层
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(),output_keep_prob=config.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)],
            state_is_tuple=True)#堆叠层数
        #初始化状态
        self._initial_state = cell.zero_state(batch_size,tf.float32)
        #因为这部分计算在gpu上没有很好的实现，因此我们使用cpu计算
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, size],dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding,input_.input_data)
        if is_training and config.keep_prob < 1: #同样为训练状态加上一层dropout
            inputs = tf.nn.dropout(inputs,config.keep_prob)
        #下面我们定义输出outputs
        outputs = []
        state = self._initial_state  #状态变量 初始化为0

        #with tf.variable_scope定义变量的作用域 对于比较复杂的模型可以很好的管理变量 提高代码的可读性

        with tf.variable_scope("RNN"):#变量作用域
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()#复用变量 每个LSTM使用同一份变量
                #inputs有三个维度第一个维度代表的是batch中第几个样本，第二个维度代表样本中第几个单词，第三个维度代表单词的向量表达维度
                #输入和输出的states为c(cell状态)和h（输出）的二元组

                #...同时传入多个样本的参数
                (cell_output,state) = cell(inputs[:,time_step,:],state)#代表所有样本的第time_step个单词
                outputs.append(cell_output)#把输出加入到outputs
        #我们将output的内容用tf.concat串到一起，并使用tf.reshape将其转化为一个很长的一维向量
        output = tf.reshape(tf.concat(outputs,1),[-1,size])
        #softmax层
        softmax_w = tf.get_variable(
            "softmax_w",[size,vocab_size],dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b",[vocab_size],dtype=tf.float32)
        logits = tf.matmul(output,softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(input_.targets,[-1])],
            [tf.ones([batch_size * num_steps],dtype=tf.float32)])
        #计算平均误差
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        #保留最终状态为final_state
        self._final_state = state
        #如果不是训练状态直接返回
        if not is_training:
            return

        #定义学习速率 并将其设为不可训练
        self._lr = tf.Variable(0.0,trainable=False)
        #返回全部可训练的参数tvars
        tvars = tf.trainable_variables()
        #针对前面得到的cost 计算tvar梯度 并设置梯度的最大范数 控制最大梯度 某种程度上起到正则化的效果
        #Gradient Explosion 可以避免梯度爆炸的问题 如果对梯度不加限制，则可能会因迭代过程中梯度过大而难以训练
        grads,_ = tf.clip_by_global_norm(tf.gradients(cost,tvars),
                                         config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        #创建训练操作 再用get_or_create_global_step生成全局统一的训练步数
        self._train_op = optimizer.apply_gradients(zip(grads,tvars),
                global_step = tf.train.get_or_create_global_step())
        #更新学习速率
        self._new_lr = tf.placeholder(
            tf.float32, shape=[],name="new_learning_rate")
        self._lr_update = tf.assign(self._lr,self._new_lr)
    #在外部修改学习速率
    def assign_lr(self,session,lr_value):
        session.run(self._lr_update,feed_dict={self._new_lr:lr_value})
    #python中的@property装饰器可以将返回变量设为只读
    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op
```

下面我们设置几种不同大小的模型参数：

```python
class SmallConfig(object):
    init_scale = 0.1 #权重值的初始scale
    learning_rate = 1.0 #学习速率的初始值
    max_grad_norm = 5 #梯度最大范数
    num_layers = 2 #LSTM可以堆叠的层数
    num_steps = 20 #LSTM梯度反向传播的展开步数
    hidden_size = 200 #隐含节点数
    max_epoch = 4 #初始学习速率可训练的epoch数
    max_max_epoch = 13 #总共可以训练的epoch数
    keep_prob = 1.0 #dropout层保留节点的比例
    lr_decay = 0.5 #学习速率衰减速率
    batch_size = 20 #每个batch中的样本数量
    vocab_size = 10000 #词汇表大小

class MediumConfig(object):
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000

class LargeConfig(object):
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000

class TestConfig(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
```

定义训练一个epoch的函数：

```python
def run_epoch(session,model,eval_op=None,verbose=False):
    # 我们记录当前时间，初始化损失costs和迭代数iters
    start_time = time.time()
    costs = 0.0
    iters = 0
    #并执行model.inittal_state来初始化状态并获得初始状态
    state = session.run(model.initial_state)
    #创建输出结果的字典表fetches
    fetches = {
        "cost":model.cost,
        "final_state":model.final_state,
    }
    #如果有评测操作也加入fetchse
    if eval_op is not None:
        fetches["eval__op"]= eval_op
    #循环训练
    for step in range(model.input.epoch_size):
        feed_dict = {}
        #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        for i,(c,h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches,feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps
        #每完成10%的epoch就进行一次结果展示
        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps"%
                  (step * 1.0 / model.input.epoch_size,np.exp(costs / iters),
                   iters * model.input.batch_size / (time.time()-start_time)))
    return np.exp(costs / iters)

```

最后我们来完成模型的主函数：

```python
raw_data = reader.ptb_raw_data('simple-examples/data/')
train_data,valid_data,test_data ,_ =raw_data

config = SmallConfig()
eval_config = SmallConfig()
eval_config.batch_size = 1
eval_config.num_steps = 1

with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)

    with tf.name_scope("Train"):
        train_input = PTBInput(config=config,data=train_data,name="TrainInput")
        with tf.variable_scope("Model",reuse=None,initializer=initializer):
            m = PTBModel(is_training=True,config=config,input_=train_input)


    with tf.name_scope("Valid"):
        valid_input = PTBInput(config=config,data=valid_data,name="ValidInput")
        with tf.variable_scope("Model",reuse=True,initializer=initializer):
            mavlid = PTBModel(is_training = False,config=config,input_=valid_input)

    with tf.name_scope("Test"):
        test_input = PTBInput(config = eval_config,data=test_data,name="TestInput")
        with tf.variable_scope("Model",reuse=True,initializer=initializer):
          mtest = PTBModel(is_training=False,config=eval_config,input_=test_input)

    sv = tf.train.Supervisor()
    with sv.managed_session() as session:
        for i in range(config.max_max_epoch):
            #学习速率衰减值
            #只需计算超过max_epoch轮数 再求超出轮数次幂
            lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch,0.0)
            #跟新学习速率
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f "% (i +1,session.run(m.lr)))
            train_perplexity = run_epoch(session,m,eval_op=m.train_op,verbose=True)
            print("Epoch: %d Train Perplexity:%.3f"%(i+1,train_perplexity))
            valid_perplexity = run_epoch(session , mavlid)
            print("Epoch: %d Valid Perplexity:%.3f"%(i+1,valid_perplexity))

        test_perplexity = run_epoch(session,mtest)
        print("Test Perplexity: %.3f" % test_perplexity)
```

最后的测试结果与论文Recurrent Neural Network Regularization的结果大致相同。模型运行成功



# 总结

LSTM模型可以储存状态，却不同于以往的RNN模型往往最后的事件产生最重要的影响。LSTM模型可以有长远的记忆，这种机制更加符合人脑的思维特点，可以模仿人来进行一些简单的记忆和推理。