# NVIDIA AI Workbench: Introduction
This is an [NVIDIA AI Workbench](https://developer.nvidia.com/blog/develop-and-deploy-scalable-generative-ai-models-seamlessly-with-nvidia-ai-workbench/) example Project that demonstrates how to customize a Stable Diffusion XL (SDXL) model. This project takes the latest SDXL model and familiarizes it with Toy Jensen via finetuning on a few pictures, thereby teaching it to generate new images which include him when it didn't recognize him previously. Next, we will also enable the user to bring their own custom image data to fine-tune the model on. Users in the [AI Workbench Early Access Program](https://developer.nvidia.com/ai-workbench-early-access) can get up and running with this Project in minutes.

## Project Description
Over the past few years Generative AI models have popped up everywhere - from creating realistic responses to complex questions, to generating images and music to impress art critics around the globe. In this project we use the Hugging Face Stable Diffusion XL (SDXL) model to create images from text prompts. You'll see how to import the SDXL model and use it to generate an image. 

From there, you'll see how you can fine-tune the model using DreamBooth. We'll use a small number of photos of Toy Jensen to fine-tune the model. This will allow us to generate new images that include Toy Jensen when the model didn't previously recognize him. After that, you'll have the chance to fine-tune the model on your own images. Perhaps you want to create an image of you at the bottom of the ocean, or in outer space? By the end of this notebook you will be able to!

## System Requirements:
* Operating System: Ubuntu 22.04
* CPU requirements: None, tested with Intel&reg; Xeon&reg; Platinum 8380 CPU @ 2.30GHz
* GPU requirements: Any NVIDIA training GPU, tested with 1x NVIDIA A100-80GB
* NVIDIA driver requirements: Latest driver version
* Storage requirements: 40GB

# Quickstart
If you have NVIDIA AI Workbench already installed, you can use this Project in AI Workbench on your choice of machine by:
1. Forking this Project to your own GitHub namespace and copying the link

   ```
   https://github.com/[your_namespace]/<project_name>
   ```
   
2. Opening a shell and activating the Context you want to clone into by

   ```
   $ nvwb list contexts
   
   $ nvwb activate <desired_context>
   ```
   
3. Cloning this Project onto your desired machine by running

   ```
   $ nvwb clone project <your_project_link>
   ```
   
4. Opening the Project by

   ```
   $ nvwb list projects
   
   $ nvwb open <project_name>
   ```
   
5. Starting JupyterLab by

   ```
   $ nvwb start jupyterlab
   ```

6. Navigate to the code directory of the project. Then, open the notebook titled ```FineTuning-SDXL.ipynb``` and get started. Happy coding!

---
**Tip:** Use ```nvwb help``` to see a full list of NVIDIA AI Workbench commands. 

---

## Tested On
This notebook has been tested with an NVIDIA A100-80gb GPU and the following version of NVIDIA AI Workbench: ```nvwb 0.5.3 (internal; linux; amd64; go1.21.3; Tue Oct 17 14:22:21 UTC 2023)```

## License
This NVIDIA AI Workbench example project is under the [Apache 2.0 License](https://github.com/nv-edwli/sdxl-customization/blob/main/LICENSE.txt)

This project will utilize additional third-party open source software projects. Review the license terms of these open source projects before use. Third party components used as part of this project are subject to their separate legal notices or terms that accompany the components. You are responsible for confirming compliance with third-party component license terms and requirements. 
