from flask import Flask, render_template, request, redirect, url_for
import time

import sys
import random
import os
from model.model import Generator
from model import utils
import torch
import torchvision.utils as vutils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
randomSeed = random.randint(1, 10000)
random.seed(randomSeed)
torch.manual_seed(randomSeed)

NZ = 100
NETG = None

def generate_image(png, cols, rows, utag):
    noises, tags = utils.fake_generator(cols * rows, NZ, device=device)
    if len(utag)!=0:
        tag = utils.get_one_hot_tag(utag)
        tag = torch.FloatTensor(tag).view(1, -1).to(device)
        tags = torch.cat([tag for _ in range(cols * rows)], dim=0)
    images = NETG(noises, tags).detach()
    path = "./generate"
    try:
        os.makedirs(path)
    except OSError:
        pass
    for i, image in enumerate(images):
        vutils.save_image(utils.denorm(image), os.path.join(path, str(i) + ".png"))
    vutils.save_image(utils.denorm(images), png, nrow=cols, padding=0)




app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    defaults = request.args.to_dict()
    hairDefault = defaults.get("hairDefault", "random")
    eyesDefault = defaults.get("eyesDefault", "random")

    if request.method == 'POST':
        form = request.form.to_dict()
        if form['generateBtn'] == 'generate':
            utag = []
            if form["hairColor"]!="random":
                utag.append(form["hairColor"])
            if form["eyesColor"]!="random":
                utag.append(form["eyesColor"])
            # print("utag:", utag)
            generate_image("static/webout.png", 10, 10, utag)          
        return redirect(url_for('home', hairDefault=form["hairColor"], eyesDefault=form["eyesColor"]))

    return render_template('home.html', 
                            imgt=str(time.time()), 
                            hairColors=utils.hair, 
                            eyesColors=utils.eyes,
                            hairDefault=hairDefault,
                            eyesDefault=eyesDefault)




if __name__ == '__main__':
    if len(sys.argv)==2:
        netGpath = sys.argv[1]
        NETG = Generator(NZ, len(utils.hair) + len(utils.eyes)).to(device)
        try:
            NETG.load_state_dict(torch.load(netGpath, map_location=lambda storage, loc: storage))
        except:
            print("`{}` not found".format(netGpath))    
        app.run(debug=True)
    else:
        print("Usage: python3 webapp.py [netG model path]")

    
 