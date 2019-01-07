# Supress default INFO logging
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
from ipywidgets import interact, widgets
from IPython.display import display, clear_output
import keywords_extractor

# get count of algorithms
empty_keywords = keywords_extractor.get_all_keywords("")
keywords_algorithm = []
for alg_name, alg_keywords in empty_keywords.items():
    for i, _ in enumerate(alg_keywords):
        keywords_algorithm.append('Keywords %s %d: ' % (alg_name, i))

l = widgets.Layout(flex='0 1 auto', height='100px', min_height='40px', width='800px')
input_text_display_widjet = widgets.HBox([widgets.Label(value='Text: '), widgets.Textarea(layout=l)])

keywords_output_widjets = [widgets.HBox([widgets.Label(value=alg), widgets.Textarea(layout=l)])
                           for alg in keywords_algorithm]
kw_rake0 = widgets.Textarea(description='RAKE algorithm 1', layout=l, disable=True)

input_text_widjet = widgets.HBox(
    [widgets.Label(value='Insert text here'), widgets.Text(layout=widgets.Layout(width='800px'))])
words_in_keyphrase = widgets.HBox(
    [widgets.Label(value='Words in keyphrase'), widgets.Dropdown(options=[1, 2, 3], value=1)])
keyphrase_length = widgets.HBox(
    [widgets.Label(value='Number of keyphrases'), widgets.IntText(value=20, disabled=False)])

out = widgets.Output()


def generate_output(w):
    input_text = w.value
    keywords = keywords_extractor.get_all_keywords(input_text, keyphrase_length.children[1].value,
                                                   words_in_keyphrase.children[1].value)
    keywords_flat = [", ".join(k) for _, v in keywords.items() for k in v]
    with out:
        clear_output(wait=True)
        input_text_display_widjet.children[1].value = input_text
        display(input_text_display_widjet)
        for i, kw_widjet in enumerate(keywords_output_widjets):
            kw_widjet.children[1].value = keywords_flat[i]
            display(kw_widjet)


def discard_all(w):
    with out:
        input_text_widjet.value = ""
        clear_output()


def keywords_params_on_change(change):
    if change['type'] == 'change' and change['name'] == 'value':
        generate_output(input_text_widjet.children[1])


input_text_widjet.children[1].on_submit(generate_output)
button = widgets.Button(description="Clear all")
button.on_click(discard_all)
keyphrase_length.children[1].observe(keywords_params_on_change)
words_in_keyphrase.children[1].observe(keywords_params_on_change)


def run():
    display(input_text_widjet)
    display(words_in_keyphrase)
    display(keyphrase_length)
    display(out)
    display(button)
