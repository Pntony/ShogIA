import json


class Encoder(json.JSONEncoder):

    def default(self, obj): # pylint: disable=E0202
        dct = obj.__dict__.copy()
        dct['class'] = str(obj.__class__).split("'")[1]
        return dct


def decoder(dct):

    module_name, class_name = dct['class'].split('.')
    module = __import__(module_name)

    if class_name == 'Game':
        obj = getattr(module, class_name)(
            dct['idx'], dct['length'], dct['is_draw'], dct['code']
        )
        return obj
    
    elif class_name == "RunManager":
        obj = getattr(module, class_name)(dct['c'])
        obj.__dict__ = dct
        return obj

    obj = getattr(module, class_name)()
    obj.__dict__ = dct
    return obj

def save(obj, file_name):
    with open(file_name, 'w') as file0:
        json.dump(obj, file0, cls=Encoder, indent=4)

def load(file_name):
    with open(file_name, 'r') as file0:
        return json.load(file0, object_hook=decoder)