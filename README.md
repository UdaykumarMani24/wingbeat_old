# Wingbeat Lab

A phone turned into a field instrument for mosquito acoustic surveillance. It
measures wingbeat frequency, runs two trained neural networks entirely on the
device, and exports labelled recordings so that a Tamil Nadu training set can
be built.

Nothing is uploaded. There is no server, no account and no network call after
the first load. It is an installable web app that works offline, because the
sites that matter most are the ones without a signal.

**Designed by Dr. Udayakumar Mani, SASTRA Deemed to be University.**

## What it does

**Measures a frequency, honestly.** Harmonic product spectrum over 20 ms
windows, with parabolic peak interpolation. The self test pushes a synthetic
450 Hz wingbeat through the same chain and recovers it to about 1 Hz, including
when the fundamental is quieter than its own harmonics.

**Runs two trained models, on the phone.** A detector that answers "is a
mosquito flying near this microphone", and a *Culex quinquefasciatus* head.
23,553 parameters each, 92 KB each, implemented in plain JavaScript with no
machine-learning library at all.

**Collects the data that does not yet exist.** Species, sex, site, temperature,
method, GPS and the raw 16-bit WAV, exported as CSV or JSON. The stored model
predictions travel with each recording, so once there are local labels it will
be possible to measure exactly where a model trained elsewhere was wrong.

## What it does not do

**It does not identify species in general.** Published field accuracy for
species classification from phone audio is roughly 35% without a location prior
and about 65% with one. Some pairs are hopeless.

**Neither model has ever heard an Indian mosquito.** Both were trained on
HumBugDB, recorded in Tanzania, Thailand, the UK, Kenya and the USA. Wingbeat
frequency follows body size and body size differs between populations. The
detector is species independent and should travel reasonably well. The *Culex*
answer is a hypothesis, not a result.

**The public data cannot support the other vectors.** It holds 22 minutes of
*Aedes aegypti*, 72 seconds of *Anopheles stephensi* and 42 seconds of *Aedes
albopictus*. That gap is measured, not assumed, and it is the clearest argument
there is for collecting local recordings.

## Verification

Every stage between the microphone and the probability is checked against the
Python original, because a model fed slightly wrong features does not crash, it
returns confident nonsense.

```
sh tools/check_all.sh
```

| Stage | Agreement with Python |
|---|---|
| resampling 44.1 kHz to 8 kHz | 0.0003 dB mean log-mel error |
| log-mel features vs librosa | 1.3e-6 dB mean |
| forward pass vs Keras | 1.8e-7 on the output probability |
| whole chain, audio to probability | 4.1e-8 |

`tools/README.md` says what these do and do not establish.

## Model accuracy

`models/report.json` is where the held-out ROC AUC, PR AUC and confusion
matrices from the training run belong. **While its fields are null, the app
states on screen that the accuracy has not been recorded.** That is deliberate:
a percentage with no provenance is read as a fact, and is worse than no number
at all. Fill it in from the output of `train_wingbeat.py`; do not estimate it.

## Running it

Any static host. For GitHub Pages, push this folder and enable Pages on the
branch root; `.nojekyll` is already present so the `models/` directory is
served. The microphone requires https, which Pages provides.

Locally:

```
python3 -m http.server 8000
```

then open `http://localhost:8000`.

## How to record so the measurement is worth keeping

Within 5 to 10 cm. Below about 50 dB of background noise. Noise suppression off,
which the app already requests from the browser, since echo cancellation,
noise suppression and automatic gain are all designed to remove exactly this
kind of faint steady tone. Confirm the species under a microscope and type it in
before saving, because a recording without an identification cannot train
anything. And note the temperature, because wingbeat frequency moves with it.

## Layout

```
index.html      the whole interface
dsp.js          FFT, harmonic product spectrum, frequency measurement
features.js     log-mel spectrogram and resampling, matched to librosa
model.js        the CNN forward pass, matched to Keras
classify.js     windowing and aggregation over a whole recording
app.js          interface, recording, storage, export
models/         the two trained models and the mel filterbank
tools/          verification scripts and their Python-generated fixtures
```
