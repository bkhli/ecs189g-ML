### Classification

To make this one work, I had to move some things around

1. MethodMLP is now initialized in the settings object. This is because it should be initialized after loading data, so that it can be initialized according to the vocab size
2. A lot of code is now running in the Data Loader object. It can't be ignored, there are potential optimizations in there.

- Text is loaded, cleaned, tokenized, and indexed there (token -> index mapping).

3. Using pytorch text is so annoying. Its deprecated, and You need to run these commands in order to make it work:

```bash
# Install older versions to make compiler happy. You could potentially tune it around to find more updated versions, but I called it a night
!pip install numpy==1.24.4
!pip install torch==2.1.0
!pip install torchtext==0.16.0
```

4. Additionally, since the datasets are binary:

- We use BCE Loss (binary cross entropy loss, and rn im using bce with logits)
- We don't need F1 score, because the classes are completely balanced (given). We could do a confusion matrix potentially?

---

##### Update Mon May 19:

Using NLTK and filtering out digits. However, this can be meaningful in reviews (if they explicity say what rating they give e.g. 8/10). Future could remedy this.

```bash
pip install nltk
# python or python3
# i don't think you need wordnet or omw-1.4, but i installed them so logging it here
# You might need punkt_tab?
python -m nltk.downloader punkt stopwords wordnet omw-1.4
```

---

### Generation

The RNN is doing okay, but I think the next step is to redo the data loader with incrementally long sequences. Right now im hard capping it to context hints of size "preview", but doing it incrementally will probably help it generate better. It goes trains through epochs quickly, but seems to take a while to actually learn. If I don't get around to doing it tomorrow and somebody wants to start, I think the most critical things to change would be:

- Setting up data incrementally (the [], the horse [], the horse walked [], and building up the whole sequence like that). It's a bit of a pain to set up the padding for the training though.
- Testing hyperparameters like hidden size, layers, learning rate, batch size etc.

##### Things to pay attention to:

1. Data loader is really important again. The tokenizing is good, but setting up the data for training good definitely be changed
2. The RNN architecture. I was spitballing when I made it, definitely change it if somehting else might work better
