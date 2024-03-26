# virtual-staging
This repository leverages stable diffusion for performing virtual staging. This goal is to build a model which can furnish empty rooms:

<img width="597" alt="image" src="https://github.com/ArvidWartenberg/virtual-staging/assets/40557722/8c042d3d-8b28-42b0-a8a3-0f3e7908bda4">

Image source: [Virtual Staging AI](https://www.virtualstagingai.app/)

The solution in this repository is not mature, and only serves as a very basic PoC.

## Approach & Delimitations
The selected approach for the virtual-staging PoC is shown in the image below:

<img width="755" alt="image" src="https://github.com/ArvidWartenberg/virtual-staging/assets/40557722/2232ceae-a8c1-4861-9506-aa56a073a7c8">


The scope was limited in the following ways since I had limited time to work on the project :
  - The scene layout representation in this work is just a binary mask.
  - The pipeline only works with unfurnished inputs, i.e., it cannot already existing furniture and generate new furniture.
  - The pipeline does not yet support custom prompts to guide style.
  - The training code does not utilize unpaired data.
Future work with regards to these points is discussed in the "outlook" section.

With these points in mind. Lets dive a bit deeper into the two modules in the figure above. The figure below shows the scene layout generator in a bit more detail.

<img width="907" alt="image" src="https://github.com/ArvidWartenberg/virtual-staging/assets/40557722/aa0b5655-878c-418c-83d2-1ec46759d5ba">


The layout generator uses two models. The initial scene generator is comprised of a ControlNet that is trained to generate the staged scenes based on the empty scene as condition to the control.
During training the prompt for the initial scene generator is extracted using an off-the shelf image captioning model on the staged scene. During inference one can infuse style and semantic information in the prompt, but for the PoC I settled on the comically simple (and bad) solution of always using the prompt "A furnished room". The initial scene is passed to OwlSam (https://huggingface.co/spaces/merve/OWLSAM), an open vocabulary instance segmentation model, together with a set of classes that we want to keep. The final generated scene layout mask is retrieved via compounding all the instance masks.

Next, the figure below shows the Inpainting ControlNet in a bit more detail.

<img width="923" alt="image" src="https://github.com/ArvidWartenberg/virtual-staging/assets/40557722/018c5734-9b80-4174-9aa0-fd054e87603f">

For this step, a "furniture agnostic" image representation of the scene is computed by masking out the pixels predicted to be "furnished" by the scene layout generator in the empty image. The ControlNet takes the agnostic image as conditional input. The prompt is given according to the same logic as for the initial scene generator.

## Training details
I only trained and evaluated using the [paired dataset](https://virtualstaging-ai.notion.site/Research-Engineer-Technical-Challenge-virtual-staging-295e57ca58b4434ebee1610bc38aa774) with 501 empty-staged samples provided by Virtual Staging AI. I randomly split the data into 491 pairs for training, and 10 for evaluation (only visual inspection)

For both the initial and second stage ControlNets I used ["v1-5-pruned.ckpt"](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt) as the pretrained latent stable diffusion "backbone", and attaching ["control_v11p_sd15_inpaint.pth"](https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_inpaint.pth
), a ControlNet that was pretrained for inpainting. This allowed me to get some results with the limited data available as the models were able to very quickly adapt to their tasks.

For compute I rented an A6000 server. I fine-tuned both controlnets on the training set for 2 epochs, using batch size 4 and lr 5e-6. This only took ~10 minutes.

## Results & Observations
The models learned their respective tasks astonishingly quickly. The image below shows how input/output pairs for the initial scene generator at the first batch, and after 100 batches.

<img width="579" alt="image" src="https://github.com/ArvidWartenberg/virtual-staging/assets/40557722/8599e1eb-0e10-43f2-80d3-4e56bc928be6">

In the image we see that initially, the model just "copy pastes" the control, but after only 100 batches it manages to output meaningful staged scenes while managing to maintain large parts of the scene structure. Similar observations were made for the second inpainting stage.

With the delimitations mentioned previously (e.g. no custom prompts, ...), below are some examples of images (alongside intermediate images/data representations) generated using the inference pipeline. These images are generated only based on the empty scenes from the validation split, and dont draw any explicit or implicit information from the corresponding staged image (e.g., not using caption extracted from the staged image).

<img width="1193" alt="Screenshot 2024-03-26 at 19 33 14" src="https://github.com/ArvidWartenberg/virtual-staging/assets/40557722/50d932dc-6839-4159-a9e3-ae6a3f269b6d">
<img width="1193" alt="Screenshot 2024-03-26 at 19 33 26" src="https://github.com/ArvidWartenberg/virtual-staging/assets/40557722/b0b1979a-2e69-4a07-ae58-a2e146db8c67">
<img width="1193" alt="Screenshot 2024-03-26 at 19 33 32" src="https://github.com/ArvidWartenberg/virtual-staging/assets/40557722/a5ec2691-3dd3-4a85-a30a-1501a7a53ba7">


Woo! The PoC sort of works. I mean, not ready for a commercial setting but the following is clear from the images above:
  - The pipeline can succeed with inpainting furniture while managing to keep information about the room.
    - While this is not perfect, you can see that the structure of the room, roof, windows, and other "rigid" parts of the room are often perserved.
  - The scene layout generator frequently succeeds in generating feasible scene layouts.
    - Moreover, it is clear that the initial generated scene image often is much worse in perserving the "rigid" parts of the scene.
    - E.g. in the first example, the initial stage generates a different roof lamp, which does not get segmented by OwlSam with our class prompts.
Of course, there are a lot of failure points in these examples, many of which are very obvious but to mention a few:
  - Unwanted changes to "geometry" of the room
  - Unrealistic/abnormal objects are generated
  - Textures and colors not properly perserved for parts not inpainted
  - ...


## Discussion & Outlook
As stated earlier, this project serves as a very simple PoC for the task of virtual staging. We have managed to show that it is feasible to solve the problem in this toy setting, but building a commercial product would put much stricter requirements, including:
  - Near perfect perservation of parts of the scene that we don't want to edit.
  - Better semantic control to ensure reasonable and realistic object shapes and combinations.
    - This point is as much about making sure that chairs don't have too few legs as it is about making sure that there is no bed in a bathroom.
  - Possibility to choose different styles for the generated images, e.g. "modern", "Scandinavian", ...

Of course, the points above can be difficult to act on directly, but some work packages I can think of are:
  - In order to improve generalization in production environments, it would be good to leverage the unpaired dataset aswell. Here the agnostic representation will let us train the second stage ControlNet without even having access to any unstaged samples.
  - Improve quality of generated staged scenes:
  -   We could leverage additional information sources in the condition to the control-net. Candidates here include: richer semantic information; e.g. specific semantic classes, "painting", "tv", "chair", etc, will let the network be more precise when inpainting. Without making any specific suggestions, investigating depth as conditional information to the ControlNet also seems promising.
  -   Improved promts: In this project the inference pipeline simply always used the prompt "a furnished room". There is tons to gain here. One can incorporate semantic- and style information in the prompt, e.g., "a modern living room with two chairs and a table", ... The challenge here is to generate the prompt based on the empty room and additional user specified input. Getting detailed semantics about the number of objects and their relation is likely difficult.

Thats the start! I really enjoyed working on this problem hope to dive deeper soon.


## How to run the code (very dirty code... and very dirty instructions...)
DISCLAIMER: I did not have time to clean the code up and make sure that it works well out-of-the-box. If you want to attempt to run this - expect that you have to solve some issues along the way.
After cloning this repo and making sure that you are in the repo root, clone ControlNet:

```git clone https://github.com/lllyasviel/ControlNet```

Next, create the environment (taken from ControlNet):

```
conda env create -f environment.yaml
conda activate control
```
For some reason I needed to also run `sudo apt-get install libsm6 libxext6 libxrender-dev` to be able to run the ControlNet code. Now, download the pretrained models and put them in the `virtual-staging/models/` folder.
```
wget -O v1-5-pruned.ckpt "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt?download=true"
wget -O control_v11p_sd15_inpaint.pth "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint.pth?download=true"
```
We can now attach the pretrained inpainting controlnet to the sd1.5 model (here you might need to do some surgery in the script to get the paths right for you):
```
python tool_add_control_hack.py 
```

Delimitation: The next step would be to generate the prompts, agnostic images and datalist. I had an issue here that there were versioning compatability issues with the `transformers` library version between ControlNet and the image captioning model that I was using. As I didn't have time to fix this, the easy way out here is to run `pip install transformers --upgrade`. If you do this you can run the data prep code:
```
python prepare_dataset.py
```

If you ran `pip install transformers --upgrade`, you have to make sure that you are back in the `control` env that we built earlier before you can run the training code. Assuming that you did this, we can now train the two models.
Here, you have to manually move out the checkpoints from the pytorch lighting logs to the models/ folder and rename them to models/masked_to_staged.ckpt, models/empty_to_staged.ckpt respectively once you are happy with your training.
```
python train.py --mode masked_to_staged
python train.py --mode empty_to_staged
```
This is all you need to run the sample inference, so lets do that!
```
python inferece.py
```
This script will simply dump images in the repo root (very clean, I know ;) ). Hopefully the results are of similar quality to what I presented!


