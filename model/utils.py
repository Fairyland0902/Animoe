import numpy as np
import numpy.random as random
import torch

hair = ['blonde hair', 'brown hair', 'black hair', 'blue hair', 'pink hair',
        'purple hair', 'green hair', 'red hair', 'silver hair', 'white hair', 'orange hair',
        'aqua hair']
hair_prob = [0.26362508351627373, 0.16645986446501862, 0.12541758136871242, 0.08933855111195953, 0.11110050587000095,
             0.12990359835830867, 0.021380166078075784, 0.03531545289682161, 0.0342655340269161, 0.018230409468359264, 0.000381788679965639,
             0.004581464159587668]
eyes = ['blue eyes', 'red eyes', 'brown eyes',
        'green eyes', 'purple eyes', 'yellow eyes', 'pink eyes', 'aqua eyes', 'black eyes']
eyes_prob = [0.40383697623365467, 0.11673188889949412, 0.10403741529063663,
             0.16588718144507014, 0.1302853870382743, 0.03102033024720817, 0.012599026438866087, 0.012694473608857497, 0.022907320797938342]
tag = ['blonde hair', 'brown hair', 'black hair', 'blue hair', 'pink hair',
       'purple hair', 'green hair', 'red hair', 'silver hair', 'white hair', 'orange hair',
       'aqua hair',
       'blue eyes', 'red eyes', 'brown eyes',
       'green eyes', 'purple eyes', 'yellow eyes', 'pink eyes', 'aqua eyes', 'black eyes']

tag_map = dict()
for i, j in enumerate(tag):
    tag_map[j] = i


def get_one_hot_tag(tags):
    one_hot = np.zeros(len(tag))
    one_hot[list(map(lambda each: tag_map[each], tags))] = 1
    return one_hot


def fake_generator(batch_size, noise_size, device):
    noise = torch.randn(batch_size, noise_size).to(device)
    hair_code = torch.zeros(batch_size, len(hair))
    eyes_code = torch.zeros(batch_size, len(eyes))
    hair_type = np.random.choice(len(hair), batch_size, p=hair_prob)
    eyes_type = np.random.choice(len(eyes), batch_size, p=eyes_prob)
    for i in range(batch_size):
        hair_code[i][hair_type[i]] = 1
        eyes_code[i][eyes_type[i]] = 1
    tags = torch.cat((hair_code, eyes_code), dim=1).to(device)

    return noise, tags


def denorm(img):
    output = img / 2 + 0.5
    return output.clamp(0, 1)
