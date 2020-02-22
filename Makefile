.PHONY: init check format

init:
	pip3 install -U pipenv
	pipenv install --dev

check:
	python3 -m isort --recursive --check-only metrics
	python3 -m black -S -l 79 --check metrics

format:
	python3 -m isort -rc -y metrics
	python3 -m black -S -l 79 metrics
