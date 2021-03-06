import logging

from .mfnet_3d import MFNET_3D, MFNET_3D_Two_Stream
from .dyn_img_net import dynimgnet50, two_stream_dynimgnet50
from .config import get_config

def get_symbol(name, use_flow, print_net=False, **kwargs):
    if name.upper() == "MFNET_3D":
        if use_flow:
            net = MFNET_3D_Two_Stream(**kwargs)
            # net = MFNET_3D(in_channels=5, **kwargs)
        else:
            net = MFNET_3D(**kwargs)
    elif name.upper() == "DYNIMGNET":
        # net = dynimgnet50(**kwargs)
        net = two_stream_dynimgnet50(**kwargs)
    else:
        logging.error("network '{}' not implemented".format(name))
        raise NotImplementedError()

    if print_net:
        logging.debug("Symbol:: Network Architecture:")
        logging.debug(net)

    input_conf = get_config(name, **kwargs)
    return net, input_conf
