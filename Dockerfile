FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

RUN pip install pipenv

# Doing this separately from ADD below reduces number of layers
# to rebuild in case deps did not change
COPY Pipfile .
COPY Pipfile.lock .

RUN pipenv install --system --deploy --ignore-pipfile

ADD . /app


# TODO
# ENTRYPOINT ["/app/docker/entrypoint.sh"]
# CMD ["daphne", "evocount_core.asgi:application"]
# CMD "/app/lupine.sh"
