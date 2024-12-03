# Installation

Please follow the guide below to install SCVIDR and its dependent software.

## Docker Image

A pre-built Docker image is available through Docker Hub.




This Docker image is built based on Mac and includes the SCVIDR software and its dependencies.

### Quick Start with Docker

In this section, we provide example commands for our SCVIDR Docker image. We assume you have basic knowledge of Docker. If not, we highly recommend learning about Docker first.

#### Step 1: Download the SCVIDR Docker image

First, download the SCVIDR Docker image from Docker Hub:

```
docker pull panda311/scvidr
```

#### Step 2: Create and run a Docker container

To run a Docker container with SCVIDR, you need to set up the ports and bind mount your local directory for easy file access. Here’s how:


1. **Run the Docker container**:

    Use the following command to run the Docker container:
    
    ```bash
    docker run -dit \
      --name scvidr_container \
      -p 8888:8888 \
      panda311/scvidr
    ```

    Here's what each part of the command does:
    
    - **`docker run`**: This command is used to create and start a new container from an image.
    - **`-dit`**: 
      - `-d` (detached mode): Runs the container in the background.
      - `-i` (interactive): Keeps STDIN open, allowing interaction with the container.
      - `-t` (tty): Allocates a pseudo-terminal, allowing you to interact with the container's shell.
    - **`--name scvidr_container`**: Assigns a custom name (`scvidr_container`) to your running container so that it can be easily identified.
    - **`-p 8888:8888`**: Maps port 8888 on your local machine to port 8888 inside the Docker container. This allows you to access Jupyter Notebook running inside the container at `http://localhost:8888`. If port 8888 is already in use, replace it with a different port (e.g., 8899) in the Docker command and access Jupyter at http://localhost:8899. Alternatively, you can copy the link provided in the terminal output after running the jupyter lab command inside the container, which will include the appropriate token for login.
    - **`panda311/scvidr`**: Specifies the Docker image to use, in this case, the `scvidr` image from Docker Hub.


#### Step 3: Access the Docker container

To enter the Docker container, run the following command:

```
docker container exec -it scvidr_container /bin/bash

```


#### Step 4: Start Jupyter Notebook

Once inside the Docker container, you can start Jupyter Notebook with the following commands:

```
jupyter lab --port=8888 --ip=0.0.0.0 --allow-root --no-browser

```

### Adding GPU Support

SCVIDR can be GPU-accelerated for faster training of models. GPU support is available on **Linux** and **Windows (via WSL 2)** systems with an NVIDIA GPU and proper drivers. Unfortunately, macOS does not support GPU passthrough for Docker containers.

#### GPU Setup for Linux

1. **Install the NVIDIA GPU drivers**:
   - Ensure you have the latest NVIDIA drivers installed for your GPU.
   
2. **Install the NVIDIA Container Toolkit**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```
  
3. **Access the Docker container with GPUS**:
  ```bash
  docker run --gpus all -dit \
  --name scvidr_gpu_container \
  -p 8888:8888 \
  panda311/scvidr
  ```

4. **Verify GPU access**: You can verify that the GPU is accessible inside the container by running.
```bash
docker exec -it scvidr_gpu_container /bin/bash
nvidia-smi
```

#### GPU Support Not Available on macOS

- **macOS** does not support GPU passthrough for Docker containers, meaning you cannot utilize NVIDIA GPUs for GPU-accelerated workloads on macOS.
- For GPU-accelerated SCVIDR usage, you’ll need to use a **Linux** system or **Windows with WSL 2**.





### Important Notes

- **WSL Performance**: If you're using Windows Subsystem for Linux (WSL), be aware that performance can be extremely slow. We don't recommend using Docker on Windows for SCVIDR.
- **Resources**: Ensure that your Docker container has enough memory to perform calculations smoothly.
- **Bugs**: If you encounter any bugs or errors with the Docker installation, please report them on the GitHub issue page.

## Docker Image Build Information

We built our Docker image using Docker's automatic build function. The Dockerfile is available [here](https://github.com/DrNamwob/scVIDR_VP_DB.git), allowing you to create a custom image if needed.
