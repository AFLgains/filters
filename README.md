
MIT License

Copyright (c) 2021 ric.porteous1989@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

# filters

## Setup

### Requirements

* Python (>=3.6)
* Sphinx

### Development environment

Create a virtual environment by running:

```shell
python -m venv .venv
```

The virtual environment should be activated every time you start a new shell session before running subsequent commands:

> On Linux/MacOS:
> ```shell
> source .venv/bin/activate
> ```

> On Windows:
> ```shell
> .venv\Scripts\activate.bat
> ```

**The above steps can also be done with some of the IDEs. eg. PyCharm <https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html>**

run:

You need few dependencies and we are using pre-commit tool to install few Git
hooks to apply basic quality check during commit/push usage.

```
pip install -r requirements.txt 
pip install pre-commit
pre-commit install
```
 
### Installation

## Usage

### in command line or in your IDE (recommanded)

```
python -m filters --help
```

### in Jupyter for exploration phase

```
jupyter notebook
```

## Configuration

All the configuration can be found in config.yml and loaded into the programm
using util/config.py

you can override per user the configuration values by adding a new file
config.$USER.yml, by default this file will not be versionned with Git.

### Docker container

```
docker build -t filters .
docker run --rm -it filters --help
```

## Test

### Unit tests

We are using [pytest](https://docs.pytest.org/en/latest/) to run
the tests

```
pytest
```

### Business tests

We are using [behave](https://behave.readthedocs.io/en/latest/) to run
the business tests

```
behave
```

## Documentation

### Generation

To generate the doc, you need to have sphinx installed then:

```
cd docs/
make html
```

### Consultation

To vizualise the documentation:

```
cd /docs/build/html/
python -m http.server
```

then open your browser to <http://0.0.0.0:8000>
