# README

![use](https://img.shields.io/badge/use-Summer%20Camp-green)
![readiness](https://img.shields.io/badge/readiness-initialization-red)

# Setup your execution environment
please refer to our Wiki [page](https://github.com/ML-for-B-E/.github/wiki/ML-Summer-Camp-courses)


# Setup your Google Cloud Storage credentials
- Contact Kévin to get a json containing the gsutil key allowing you to read the bucket EEIA
- Store it somewhere for instance `~/gcp/auth.json` and get the path
- store it in an environment variable `export GOOGLE_APPLICATION_CREDENTIALS=~/gcp/auth.json`. Better to put the line in your `.bashrc` to persist the key for every terminal.
