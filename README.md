# virtual-staging
This repository leverages stable diffusion for performing virtual staging. This goal is to build a model which can furnish empty rooms:
<img width="597" alt="image" src="https://github.com/ArvidWartenberg/virtual-staging/assets/40557722/8c042d3d-8b28-42b0-a8a3-0f3e7908bda4">

Image source: Virtual Staging AI (https://www.virtualstagingai.app/)

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


## Discussion
The logic behond the two-stage approach is that the first-stage inpainting might add and/or replace more semantic information in the original scene than we wish. By exctracting semantic information based on certain object classes as an intermediate step, we can conduct the second furnishing stage in a way where we have more semantic conditional control such that we can avoid painting features that we do not want.



