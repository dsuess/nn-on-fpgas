# This makefile has been created to help developers perform common actions.
# Most actions assume it is operating in a virtual environment where the
# python command links to the appropriate virtual environment Python.

# Do not remove this block. It is used by the 'help' rule when
# constructing the help output.
# help:
# help: classilvier Makefile help
# help:

# help: help                           - display this makefile's help information
.PHONY: help
help:
	@grep "^# help\:" Makefile | grep -v grep | sed 's/\# help\: //' | sed 's/\# help\://'

# help: init                          - create an environment for development
.PHONY: init
init:
	@poetry run pip install -U pip
	@poetry install

# help: clean                          - clean all files using .gitignore rules
.PHONY: clean
clean:
	@git clean -X -f -d

# help: scrub                          - clean all files, even untracked files
.PHONY: scrub
scrub:
	@git clean -x -f -d

# help: test                           - run tests
.PHONY: test
test:
	@poetry run pytest fcnn-mnist utils

# help: style                          - perform code formatting
.PHONY: style
style:
	@poetry run isort frcnn-mnist utils
	@poetry run black --include .py --exclude ".pyc|.pyi|.so" fcnn-mnist utils

# help: check                          - perform linting checks
.PHONY: check
check:
	@poetry run isort --check fcnn-mnist utils
	@poetry run black --check --include .py --exclude ".pyc|.pyi|.so" frcnn-mnist utils

# Keep these lines at the end of the file to retain nice help
# output formatting.
# help:
