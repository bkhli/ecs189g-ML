

### Classification

To make this one work, I had to move some things around
1) MethodMLP is now initialized in the settings object. This is because it should be initialized after loading data, so that it can be initialized according to the vocab size
2) A lot of code is now running in the Data Loader object. It can't be ignored, there are potential optimizations in there.
- Text is loaded, cleaned, tokenized, and indexed there (token -> index mapping).
3) Using pytorch text is so annoying. Its deprecated, and You need to run these commands in order to make it work:
```bash
# Install older versions to make compiler happy. You could potentially tune it around to find more updated versions, but I called it a night
pip install torch==2.1.0
pip install torchtext==0.16.0
pip install numpy==1.24.4
```
4) Additionally, since the datasets are binary:
- We use BCE Loss (binary cross entropy loss, and rn im using bce with logits)
- We don't need F1 score, because the classes are completely balanced (given). We could do a confusion matrix potentially?

---

##### Update Mon May 19:
Using NLTK and filtering out digits. However, this can be meaningful in reviews (if they explicity say what rating they give e.g. 8/10). Future could remedy this.
