import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.tools.graph_transforms import TransformGraph
#from tensorflow.keras import backend as K
import os
import tensorrt as trt
import onnx
import onnx.backend as backend
import logging



TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def freeze_and_optimize_session(session, keep_var_names=None, input_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        graph = tf.graph_util.remove_training_nodes(
            input_graph_def, protected_nodes=output_names)
        graph = tf.graph_util.convert_variables_to_constants(
            session, graph, output_names, freeze_var_names)
        transforms = [
          'remove_nodes(op=Identity)',
          'merge_duplicate_nodes',
          'strip_unused_nodes',
          'fold_constants(ignore_errors=true)',
          'fold_batch_norms',
         ]
        graph = TransformGraph(
            graph, input_names, output_names, transforms)
        return graph


def h5_to_pb(folder , model_name):
  # freeze Keras session - converts all variables to constants
  tf.compat.v1.keras.backend.set_learning_phase(0)
  print('Model-path -> ', folder +'/'+ model_name + ".h5")
  model = load_model(folder +'/'+ model_name + ".h5", custom_objects=None)

  graph_before = tf.compat.v1.keras.backend.get_session().graph
  print('input : -> ', [inp.op.name for inp in model.inputs])
  print('output: -> ', [out.op.name for out in model.outputs])
  frozen_graph = freeze_and_optimize_session(tf.compat.v1.keras.backend.get_session(),
                         input_names=[inp.op.name for inp in model.inputs],
                         output_names=[out.op.name for out in model.outputs])
  tf.io.write_graph(frozen_graph,
             logdir=folder,
             as_text=False,
             name= model_name + '.pb')
  
  # To check graph in text editor 
  ### IF YOU WANT TO USE TENSORBOARD - SAVE AS TEXT IN FORMAT .PBTXT
  # tf.io.write_graph(frozen_graph,
  #           logdir=folder,
  #           as_text=True,
  #           name=model_name+'.pbtxt')
  

def pb_to_onnx(folder, model_name):
    # pb -> onnx
    os.system("python3 -m tf2onnx.convert --graphdef {}.pb --output {}.onnx --inputs input_2:0 --outputs probabilistic_output/Sigmoid:0,positional_output/Reshape:0 --opset=11 ".format(folder +'/'+ model_name, folder +'/' + model_name))




def network_structure(args):
    model_path = args['model']
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        output_graph_def = tf.GraphDef()
        # Get the default picture
        graph = tf.get_default_graph()
        with open(model_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
            # Get how many operation nodes in the current graph
            print("%d ops in the graph." % len(output_graph_def.node))
            op_name = [tensor.name for tensor in output_graph_def.node]
            print(op_name)
            print('=======================================================')
            # Produce log files in the log_graph folder, you can visualize the model in tensorboard
            summaryWriter = tf.summary.FileWriter('log_graph_'+args['model'], graph)
            cnt = 0
            print("%d tensors in the graph." % len(graph.get_operations()))
            for tensor in graph.get_operations():
                # print out the name and value of tensor
                print(tensor.name, tensor.values())
                cnt += 1
                if args['n']:
                    if cnt == args['n']:
                        break


def add_profiles(config, inputs, opt_profiles):
    logger.debug("=== Optimization Profiles ===")
    for i, profile in enumerate(opt_profiles):
        for inp in inputs:
            _min, _opt, _max = profile.get_shape(inp.name)
            logger.debug("{} - OptProfile {} - Min {} Opt {} Max {}".format(inp.name, i, _min, _opt, _max))
        config.add_optimization_profile(profile)


def create_optimization_profiles(builder, inputs, batch_sizes=[1]): 
    # Check if all inputs are fixed explicit batch to create a single profile and avoid duplicates
    if all([inp.shape[0] > -1 for inp in inputs]):
        profile = builder.create_optimization_profile()
        for inp in inputs:
            fbs, shape = inp.shape[0], inp.shape[1:]
            profile.set_shape(inp.name, min=(fbs, *shape), opt=(fbs, *shape), max=(fbs, *shape))
            return [profile]
    
    # create several profiles
    profiles = {}
    for bs in batch_sizes:
        if not profiles.get(bs):
            profiles[bs] = builder.create_optimization_profile()

        for inp in inputs: 
            shape = inp.shape[1:]
            # Check if fixed explicit batch
            if inp.shape[0] > -1:
                bs = inp.shape[0]
            profiles[bs].set_shape(inp.name, min=(bs, *shape), opt=(bs, *shape), max=(bs, *shape))

    return list(profiles.values())


def onnx_to_trt(folder, model_name):
    print('--- fp_16 ---')

    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    F = EXPLICIT_BATCH
    
    NUM_IMAGES_PER_BATCH = 1

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(F) as network,trt.OnnxParser(network, TRT_LOGGER) as parser, builder.create_builder_config() as config:
        if builder.platform_has_fast_fp16:
            print("Jetson has fast fp16 mode")
       
        builder.max_batch_size = NUM_IMAGES_PER_BATCH
        builder.max_workspace_size = 1 << 32
        builder.fp16_mode = True
        builder.strict_type_constraints = True

        config.max_workspace_size = 1 << 32
        config.flags |= 1 << int(trt.BuilderFlag.FP16)
        config.flags |= 1 << int(trt.BuilderFlag.STRICT_TYPES)

        with open("./{}/{}.onnx".format(folder, model_name), 'rb') as model:
            PARSED = parser.parse(model.read())
            if not PARSED:
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            else:
                for i in network:
                    print(i.name)

                inputs = [network.get_input(i) for i in range(network.num_inputs)]
                #print('inputs => ', inputs)
                opt_profiles = create_optimization_profiles(builder, inputs)
                add_profiles(config, inputs, opt_profiles)
                
                engine = builder.build_engine(network, config)
            with open('./{}/{}.fp16.engine'.format(folder, model_name), "wb") as engine_file:
                engine_file.write(engine.serialize())
    return engine


if __name__ == "__main__":
    
    folder = 'weights/converted'
    model_name = 'model_classes8'
    
    # args = {'model':'converted/model_classes8.pb',
    #         'n' : 200}
    # network_structure(args)
    
    try:
       h5_to_pb(folder, model_name)
    except Exception as e:
        print('\n\nError: h5_to_pb')
        print(e)

    try:
       pb_to_onnx(folder, model_name)
    except Exception as e:
        print('\n\nError: pb_to_onnx')
        print(e)

    try:
       onnx_to_trt(folder, model_name)
    except Exception as e:
        print('\n\nError: onnx_to_trt')
        print(e)
