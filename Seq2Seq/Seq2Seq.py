import tensorflow as tf
from tensorflow.python.layers.core import Dense

class Seq2Seq_Model:
    """
    This class helps to create a seq2seq model of desired configurations.
    """
    def __init__(self,params):
        """
        Mention the parameters required to create Seq2Seq model
        Arguments:
        cell_type - can be any of LSTM, GRU, RNN, GLSTM
        num_units - number of cells each layer should have
        num_encoder_layers - number of stacked encoder layers to use
        num_decoder_layers - number of stacked decoder layers to use
        embeddings - matrix of embeddings for words
        word2int - dictionary mapping words to their ids
        attention_machanism - type of attention to use. can be None, Luong, Bahdanau, LuongScaled, BahdanauNormalised
        bidirectional - boolean value to mention whether to use bidirectional training
        inference_mechanism - type of inference mechanism to use. can be Beam or Greedy
        """
        self.cell_type = params['cell_type']
        self.num_units = params['num_units']
        self.enc_layers = params['num_encoder_layers']
        self.dec_layers = params['num_decoder_layers']
        self.embeddings = params['embeddings']
        self.word2int = params['word2int']
        self.vocab_size = params['vocab_size']
        self.beam_size = params['beam_size']
        self.end_token = params['end_token']
        self.attn_mech = params['attention_mechanism']
        self.bidirection = params['bidirectional']
        self.infer_mech = params['inference_mechanism']
        self.batch_size = params['batch_size']
    
    def _init_placeholders_(self):
        """
        Initialize placeholders for the model
        """
        self.encoder_inputs = tf.placeholder(dtype=tf.int32,shape=(None,None),name='Encoder_Inputs')
        self.decoder_targets = tf.placeholder(dtype=tf.int32,shape=(None,None),name='Decoder_Targets')
        self.encoder_lengths = tf.placeholder(dtype=tf.int32,shape=(None,),name='Encoder_Lengths')
        self.decoder_lengths = tf.placeholder(dtype=tf.int32,shape=(None,),name='Decoder_Lengths')
        self.max_dec_length = tf.reduce_max(self.decoder_lengths, name='Max_Decoder_Length')
        self.learning_rate = tf.placeholder(dtype=tf.float32,name='Learning_Rate')
    
    def _single_rnn_(self,num_units):
        """
        Create a single rnn layer of given cell type and number of units
        """
        cell_type = self.cell_type.upper()
        if cell_type == 'LSTM':
            return tf.contrib.rnn.LSTMCell(num_units,
                                           initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2))
        elif cell_type == 'GRU':
            return tf.contrib.rnn.GRUCell(num_units,
                                          initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2))
        elif cell_type == 'GLSTM':
            return tf.contrib.rnn.GLSTMCell(num_units,
                                            initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2))
        
    def _multi_rnn_(self,num_units,num_layers):
        """
        Creates a list of rnn layers of mentioned cell type and number of layers
        """
        return [self._single_rnn_(num_units) for _ in range(num_layers)]
    
    def _simple_encoder_(self):
        """
        Construct Simple Encoder Network
        """
        enc_embed_input = tf.nn.embedding_lookup(self.embeddings,self.encoder_inputs)
        if self.enc_layers == 1:
            cells = self._single_rnn_(self.num_units)
            self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(cell=cells,
                                                                               inputs=enc_embed_input,
                                                                               sequence_length=self.encoder_lengths,
                                                                               dtype=tf.float32)
        else:
            cells = tf.contrib.rnn.MultiRNNCell(cells=self._multi_rnn_(self.num_units,self.enc_layers))
            self.encoder_outputs, encoder_states = tf.nn.dynamic_rnn(cell=cells,
                                                                     inputs=enc_embed_input,
                                                                     sequence_length=self.encoder_lengths,
                                                                     dtype=tf.float32)
            self.encoder_final_state = encoder_states[-1]
    
    def _bidirectional_encoder_(self):
        """
        Contruct Bidirectional Encoder Network 
        """
        enc_embed_input = tf.nn.embedding_lookup(self.embeddings,self.encoder_inputs)
        if self.enc_layers == 1:
            cells_fw = self._single_rnn_(self.num_units)
            cells_bw = self._single_rnn_(self.num_units)
            outputs, (fw_states, bw_states) = tf.nn.bidirectional_dynamic_rnn(cells_fw=cells_fw,
                                                                              cells_bw=cells_bw,
                                                                              inputs=enc_embed_input,
                                                                              sequence_length=self.encoder_lengths,
                                                                              dtype=tf.float32)
            self.encoder_outputs = tf.concat(outputs,axis=2,name="Encoder_Outputs_Concat")
            c = tf.concat((fw_states[0].c,bw_states[0].c),axis=1,name='Encoder_C_State_Concat')
            h = tf.concat((fw_states[0].h,bw_states[0].h),axis=1,name='Encoder_H_State_Concat')
            self.encoder_final_state = tf.contrib.rnn.LSTMStateTuple(c,h)
        else:
            cells_fw = self._multi_rnn_(self.num_units,self.enc_layers)
            cells_bw = self._multi_rnn_(self.num_units,self.enc_layers)
            outputs, fw_states, bw_states = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw,
                                                                                           cells_bw=cells_bw,
                                                                                           inputs=enc_embed_input,
                                                                                           sequence_length=self.encoder_lengths,
                                                                                           dtype=tf.float32)
            self.encoder_outputs = tf.concat(outputs,axis=2,name="Encoder_Outputs_Concat")
            c = tf.reduce_mean([tf.concat((fw_states[i].c,bw_states[i].c),axis=1) 
                                for i in range(self.enc_layers)],axis=0,name="Encoder_C_State_Concat")
            h = tf.reduce_mean([tf.concat((fw_states[i].h,bw_states[i].h),axis=1) 
                                for i in range(self.enc_layers)],axis=0,name="Encoder_H_State_Concat")
            self.encoder_final_state = tf.contrib.rnn.LSTMStateTuple(c,h)
                
    def _decoder_without_attention_(self):
        """
        Creates a Decoder Network Without Attention Mechanism
        """
        self.output_layer = Dense(units=self.vocab_size,
                             kernel_initializer=tf.truncated_normal_initializer(mean = 0.0, stddev=0.1),
                             name='Output')
        # bidirectional encoder gives twice the outputs size. So decoder units need to be double the size
        if self.bidirection:
            num_units = self.num_units*2
        else:
            num_units = self.num_units
        if self.dec_layers == 1:
            cells = self._single_rnn_(num_units)
        else:
            cells = self._multi_rnn_(num_units,self.dec_layers)
        initial_state = [self.encoder_final_state for _ in range(self.dec_layers)]
        initial_state[0] = cells[0].zero_state(dtype=tf.float32,batch_size=self.batch_size)
        self.decoder_initial_state = tuple(initial_state)
        self.dec_cells = tf.contrib.rnn.MultiRNNCell(cells)
        
    def _decoder_with_attention_(self,memory, memory_lengths):
        """
        Creates a Decoder Network With Attention Mechanism 
        """
        self.output_layer = Dense(units=self.vocab_size,
                             kernel_initializer=tf.truncated_normal_initializer(mean = 0.0, stddev=0.1),
                             name='Output')
        # bidirectional encoder gives twice the outputs size. So decoder units need to be double the size
        if self.bidirection:
            num_units = self.num_units*2
        else:
            num_units = self.num_units
        if self.dec_layers == 1:
            cells = self._single_rnn_(num_units)
        else:
            cells = self._multi_rnn_(num_units,self.enc_layers)
    
        self.attn_mech = self.attn_mech.lower()
        if self.attn_mech == 'bahdanau':
            attention = tf.contrib.seq2seq.BahdanauAttention(memory=memory,
                                                             num_units=num_units,
                                                             memory_sequence_length=memory_lengths,
                                                             name=self.attn_mech)
        elif self.attn_mech == 'luong':
            attention = tf.contrib.seq2seq.LuongAttention(memory=memory,
                                                          num_units=num_units,
                                                          memory_sequence_length=memory_lengths,
                                                          name=self.attn_mech)
        elif self.attn_mech == 'bahdanaunormalized':
            attention = tf.contrib.seq2seq.BahdanauAttention(memory=memory,
                                                             num_units=num_units,
                                                             memory_sequence_length=memory_lengths,
                                                             normalize=True,
                                                             name=self.attn_mech)
        elif self.attn_mech == 'luongscaled':
            attention = tf.contrib.seq2seq.LuongAttention(memory=memory,
                                                          num_units=num_units,
                                                          memory_sequence_length=memory_lengths,
                                                          scale=True,
                                                          name=self.attn_mech)
        
        cells[0] = tf.contrib.seq2seq.AttentionWrapper(cell=cells[0],
                                                       attention_mechanism=attention,
                                                       attention_layer_size=self.num_units)
        
        initial_state = [self.encoder_final_state for _ in range(self.dec_layers)]
        attention_cell_state = cells[0].zero_state(dtype=tf.float32,batch_size=self.batch_size)
        initial_state[0] = attention_cell_state.clone(cell_state=initial_state[0])
        self.decoder_initial_state = tuple(initial_state)
        self.dec_cells = tf.contrib.rnn.MultiRNNCell(cells)
        
    def _training_decoder_(self):
        ending = tf.strided_slice(self.decoder_targets,begin=[0,0],end=[self.batch_size,-1],strides=[1,1])
        self.decoder_inputs = tf.concat([tf.fill([self.batch_size, 1],self.word2int['<GO>']),ending],1)
        dec_embed_input = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)
        
        train_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,sequence_length=self.decoder_lengths,name="Training_Helper")
        train_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.dec_cells,
                                                        helper=train_helper,
                                                        initial_state=self.decoder_initial_state,
                                                        output_layer=self.output_layer)
        self.training_logits,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder=train_decoder,
                                                                     impute_finished=True,
                                                                     swap_memory=True,
                                                                     maximum_iterations=self.max_dec_length)
        
    def _inference_decoder_(self):
        ending = tf.strided_slice(self.decoder_targets,begin=[0,0],end=[self.batch_size,-1],strides=[1,1])
        self.decoder_inputs = tf.concat([tf.fill([self.batch_size, 1],self.word2int['<GO>']),ending],1)
        dec_embed_input = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)
        start_tokens = tf.fill([self.batch_size], self.word2int['<GO>'])
        if self.infer_mech == 'beam':
            infer_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=self.dec_cells,
                                                                 embedding=self.embeddings,
                                                                 start_tokens=start_tokens,
                                                                 end_token=self.word2int[self.end_token],
                                                                 initial_state=self.decoder_initial_state,
                                                                 beam_width=self.beam_size,
                                                                 output_layer=self.output_layer)
        else:
            infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embeddings,
                                                                     start_tokens=start_tokens,
                                                                     end_token=self.word2int[self.end_token])
            infer_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.dec_cells,
                                                            helper=infer_helper,
                                                            initial_state=self.decoder_initial_state,
                                                            output_layer=self.output_layer)
        final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(infer_decoder,
                                                                impute_finished=True,
                                                                swap_memory=True,
                                                                maximum_iterations=self.max_dec_length)
        if self.infer_mech == 'beam':
            beam_predictions = final_outputs.predicted_ids
            self.inference_logits = tf.transpose(beam_predictions, perm=[0, 2, 1])
        else:
            self.inference_logits = tf.identity(final_outputs.sample_id, name='predictions')
    
    def optimization(self):
        training_logits = tf.identity(self.training_logits.rnn_output, 'logits')
        masks = tf.sequence_mask(self.decoder_lengths,self.max_dec_length,dtype=tf.float32,name='masks')
        cost = tf.contrib.seq2seq.sequence_loss(training_logits,self.decoder_targets,masks)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
        
    def build_graph(self,infer=False):
        graph = tf.Graph()
        with graph.as_default():
            with tf.variable_scope("Inputs",reuse=tf.AUTO_REUSE):
                self._init_placeholders_()
            with tf.variable_scope("Encoder_Network",reuse=tf.AUTO_REUSE):
                if self.bidirection:
                    self._bidirectional_encoder_()
                else:
                    self._simple_encoder_()
            with tf.variable_scope("Decoder_Network",reuse=tf.AUTO_REUSE):
                if not infer:
                    if self.attn_mech == None:
                        self._decoder_without_attention_()
                    else:
                        memory = self.encoder_outputs
                        memory_lengths = self.encoder_lengths
                        self._decoder_with_attention_(memory,memory_lengths)
                    self._training_decoder_()
                    with tf.name_scope("optimization"):
                        self.optimization()
                else:
                    if self.attn_mech == None:
                        self._decoder_without_attention_()
                    else:
                        if self.infer_mech == 'beam':
                            memory = tf.contrib.seq2seq.tile_batch(self.encoder_outputs, self.beam_size)
                            memory_sequence_length = tf.contrib.seq2seq.tile_batch(self.encoder_lengths, self.beam_size)
                            self.batch_size = self.batch_size*self.beam_size
                            self.encoder_final_state = tf.contrib.seq2seq.tile_batch(self.encoder_final_state,self.beam_size)
                        else:
                            memory = self.encoder_outputs
                            memory_lengths = self.encoder_lengths
                        self._decoder_with_attention_(memory,memory_lengths)
                    self._inference_decoder_()
        return graph