# Scraping Frequently Asked Questions

This subdirectory deals with webscraping Frequently Asked Questions (FAQs). The goal is to scrape data from trusted source and store the data in our [schema](https://github.com/JHU-COVID-QA/scraping-qas/wiki/Schema-v0.3). Another group will then deal with the NLP to make this data useful.

## Setup
You will need anaconda3 to be able to work on this project. anaconda3 is avalible [here](https://www.anaconda.com/products/individual). Once anaconda is installed, you can clone this repo with `git clone https://github.com/JHU-COVID-QA/scraping-qas.git`

Once the repo is cloned change your working directory to `scraping-qas/src/scraping/` by running `cd scraping-qas/src/scraping/`

Run `conda env create -f environment.yml` to setup the conda environment with the correct configurations.
This project uses python3.6. 

Make sure to then run `python setup.py install`. This will create a local library called `covid_scraping` that you will use.

Finally run `conda activate crawler` to activate the virtual enviroment

### Installing dependenceis

Use conda to install dependencies

### Updating the conda env

If you installed new dependencies, run `conda env export --from-history --ignore-channel > environment.yml.tmp`.
Now, merge `environment.yml.tmp` and `environment.yml` into `environment.yml` so that you do not overwrite other
dependencies in the yml file.
Finally, push the new `environment.yml` configuration file

If `enviroment.yml` has changed you can update your virtual enviroment with `conda env update  --file environment.yml  --prune`


## Websites to scrape

We have a list of websites to scrape. Please choose one of the websites from our [todo list](https://github.com/JHU-COVID-QA/scraping-qas/projects/1).

### Scraping a new website
Once you have claimed a website to work on, move it from the To scrape column to the Scraper work in progress column on our [board](https://github.com/JHU-COVID-QA/scraping-qas/projects/1) and assign yourself to the issue corresponding to the website.

Next, create a new branch using
`git checkout -b <name-of-new-branch>` where the branch name should be `scraping-<name of website>-<issue number>` Where `name of website` is the organization that runs the site you are scraping, and `issue number` is the issue number associated with the Github issue for that site.
You will implement your scraper in a new file in https://github.com/JHU-COVID-QA/scraping-qas/tree/master/src/scraping/scrapers.
Please name the new file the name of the website you are scraping, so if you are scraping FAQs from the World Health Organization, the filename should be `who.py`. 

#### Implementing Scraper class
All your code needs to do is implement the [Scraper abstract class](https://github.com/JHU-COVID-QA/scraping-qas/blob/f3383db720cc451ad10b60bd6eca07d820658e46/src/scraping/covid_scraping/scraper.py#L16-L30).

Look at [example_scraper](https://github.com/JHU-COVID-QA/scraping-qas/blob/master/src/scraping/scrapers/example_scraper.py) on how to implement the `scrape()` function.

The `scrape()` function should use the [Conversion class](https://github.com/JHU-COVID-QA/scraping-qas/blob/f3383db720cc451ad10b60bd6eca07d820658e46/src/scraping/covid_scraping/conversion.py#L20) which converts the data into our schemas. 

The conversion classes constructor takes in two arguments, `file_prefix` and `path`.  `file_prefix` and `path` are members of the scraper class. So and a constructor for conversion class would be
```python
converter = Conversion(self._filename, self._path)
```

Then for each question answer pair that you scrape, you need to add it to the conversion class. The conversion class expects a dictionary with serveral fields populated. An example of adding an example to the conversion class would be.

```python
converter.addExample({
    'sourceUrl': 'example.com',
    'sourceName': "example",
    "needUpdate": True,
    "typeOfInfo": "QA",
    "isAnnotated": False,
    "responseAuthority": "",
    "question": '<a href="example.com/dir1">What is COVID-19?</a>',
    "answer": '<p><a href="example.com/dir2">Coronaviruses</a> are a large family of viruses.</p>',
    "hasAnswer": True,
    "targetEducationLevel": "NA",
    "topic": ['topic1', 'topic2'],
    "extraData": {'example extra field': 'example value'},
    "targetLocation": "US",
    "language": 'en',
})
```
Finally the last thing you need to do is return the value of the `write()` fucntion. You do this with 
```python
return converter.write()
```

#### Code styling
Before you are finished, make sure that your code abides by our coding style. We use standard [pep8](https://www.python.org/dev/peps/pep-0008/). Run `pep8 <python file name>`. Please fix all style comments (except for line length, and "module level import not at top of file").

Now your scraper should be done, and its time to make a pull request.

## Pull requests
Once you get to this point, please push your scraper to the branch you created with `git push origin <branch-name>` and then go the github pull request [page](https://github.com/JHU-COVID-QA/scraping-qas/pulls) to make a pull request and assign the pull request to @azpoliak.

## Bibliography 

If you use the dataset, please cite us using the following citation

```
@inproceedings{Collecting+COVID_NLP20202,
    title={Collecting Verified COVID-19 Question Answer Pairs},
  author={Poliak, Adam and Fleming, Max and Costello, Cash and Murray, Kenton W and Yarmohammadi, Mahsa and Pandya, Shivani and Irani, Darius and Agarwal, Milind and Sharma, Udit and Sun, Shuo and Ivanov, Nicola and Shang, Lingxi and Srinivasan, Kaushik and Lee, Seolhwa and Han, Xu and Agarwal, Smisha and Sedoc, João},
  year={2020},
  booktitle={NLP COVID-19 Workshop @EMNLP},
  url={https://openreview.net/forum?id=GR03UfD2OZk}
}
```

