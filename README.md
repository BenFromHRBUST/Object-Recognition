# Object Recognition

## Contents
- [introduction](#introduction)
- [How to use](#How-to-use)
- [Version Management](#version-management)
- [Authors](#Authors)
- [Copyright](#Copyright)

### Introduction

[//]: # (TODO: Write a project description)

### How to use

#### Download the project
```bash
git clone https://github.com/BenFromHRBUST/Object-Recognition.git
```

#### Install the dependencies (not available yet)

[//]: # (TODO: add requirements.txt)

```bash
pip install -r requirements.txt
```

#### Configure the project
To protect possible privacy settings, you need to create your own `config.py` file. You can copy the `config.template.py` file and rename it to `config.py`. Then you can configure the project according to your own needs.
```bash
# Copy the config.template.py file and rename it to config.py
cp config.template.py config.py

# Edit the config.py file using your favorite editor (e.g. vim)
vim config.py
```

If you want to use wandb, you need to set the `api_key` in the config.py file. It will only be used when you set the `production` to True.

#### Run the project
```bash
python main.py
```

### version management

This project uses [Git](https://git-scm.com/) for version management. You can see the currently available versions in the repository.

### Authors
- Name: Hongyi Huang
- Email: hongyi.huang-3@postgrad.manchester.ac.uk
- Github: [BenFromHRBUST](https://github.com/BenFromHRBUST)

### Copyright

This project is licensed under the MIT license. For details, please see [LICENSE.txt](https://github.com/shaojintian/Best_README_template/blob/master/LICENSE.txt)
