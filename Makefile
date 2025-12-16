# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mezhang <mezhang@student.42heilbronn.de    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/12/16 16:08:37 by mezhang           #+#    #+#              #
#    Updated: 2025/12/16 16:45:27 by mezhang          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

install:
	@echo "Installing project dependencies..."
	uv sync

run:
	@echo "Running the project..."
	${PYTHON_CMD} -m ${SRC_DIR}

debug:
	@echo "Debugging..."
	${PYTHON_CMD} -m pdb -m ${SRC_DIR}

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__
	rm -rf ${SRC_DIR}/__pycache__
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache

lint:
	@echo "Running flake8 linter..."
	uv run flake8 .
	@echo "Running mypy type checker..."
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@echo "Running flake8 linter with strict settings..."
	uv run flake8 .
	@echo "Running mypy type checker with strict settings..."
	uv run mypy . --strict