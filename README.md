*This project has been created as part of the 42 curriculum by amarquez*

# Call Me Maybe

## Description

A function calling system: it translates a natural language request into a
structured function call, without answering the request itself.

Given `What is the sum of 2 and 3?`, the program does not return `5`. It returns
the function to call and its typed arguments:

```json
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
        "a": 2.0,
        "b": 3.0
    }
}
```

Small models such as `Qwen/Qwen3-0.6B` are unreliable at producing structured
output from a prompt alone. This project uses **constrained decoding**: before
each token is chosen, every token that would break the JSON structure or the
function's schema is forbidden. The model still decides which function fits and
which values to use, but it cannot produce invalid output.

## Instructions

### Installation

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14 or later.

```sh
git clone <repository-url> call_me_maybe
cd call_me_maybe
uv sync
```

The model weights are downloaded from Hugging Face on the first run.

### Running

```sh
make run
```

By default, this reads `data/input/functions_definition.json` and
`data/input/function_calling_tests.json`, then writes
`data/output/function_calling_results.json`. All three paths can be changed:

```sh
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

### Other commands

| Command | What it does |
|---|---|
| `make install` | Installs the dependencies (`uv sync`) |
| `make debug` | Runs the program under `pdb` |
| `make test` | Runs the tests with coverage |
| `make lint` | Runs `flake8` and `mypy` with the required flags |
| `make lint-strict` | Runs `flake8` and `mypy --strict` |
| `make clean` | Removes the local caches |
| `make fclean` | Also removes the virtual environment and the **whole** uv and Hugging Face caches |

## Algorithm Explanation

The output has a known shape, so it is split into fixed text, written by the
program, and values, chosen by the model:

```text
{"prompt": "Greet shrek", "name": "fn_greet", "parameters": {"name": "shrek"}}
└──────────── fixed ─────────────┘└─ model ┘└──────── fixed ────────┘└ model ┘
```

Fixed text costs no model call and cannot be wrong.

**Masking.** At each step the model scores every token in the vocabulary. The
decoder sets the score of every illegal token to negative infinity and takes the
highest remaining score, so an illegal token can never be chosen.

**Choosing the function.** All function names are tokenized up front. The model
is only called where the remaining names differ, and its choice is limited to
their next tokens. Shared tokens such as `fn_`, and the rest of a name once it is
the only one left, are added without a model call.

**Generating values.** Each parameter type has a constraint with three methods:
`allowed(buf)`, `is_complete(buf)` and `value(buf)`, where `buf` holds the bytes
written so far for that value.

| Type | Rules |
|---|---|
| `number`, `integer` | A state machine over `-?digits(.digits)?` that ends with `,` or `}` |
| `string` | Valid UTF-8 text with no `"`, `\` or control characters, ended by `"` |
| `boolean` | Only `true` or `false` |

**Ending strings.** The model usually closes a string with a combined token such
as `",` or `"}`. These count as one group: the bare `"` takes the best score in
the group, and only `"` can be chosen. The program then writes the separator or
closing braces itself.

**Length cap.** After 48 tokens in a value, only its closing token stays legal,
unless the value cannot end yet (such as `-` or `12.`). In that case it continues
until it can.

**Final result.** The result is built from typed values rather than by parsing
generated text, and is then validated by a pydantic model.

## Design Decisions

```text
┌──────────────────────────────────────────────┐
│  __main__.py                                 │
│  Orchestration: read files, loop prompts,    │
│  write results.                              │
└──────────────────────────────────────────────┘
          │                     │
          ▼                     ▼
     ┌────────┐         ┌──────────────┐
     │ loader │         │ decoder      │
     │        │         │              │
     │ file → │         │ generation   │
     │ model  │         │ loop         │
     └────────┘         └──────────────┘
                                │
                    ┌───────────┴────────┐
                    ▼                    ▼
             ┌──────────────┐   ┌──────────────┐
             │ context      │   │ constraint   │
             │              │   │              │
             │ schema +     │   │ generated    │
             │ prompt → LLM │   │ text → legal │
             │ input text   │   │ token IDs    │
             └──────────────┘   └──────────────┘
                                       │
                                       ▼
                                ┌──────────────┐
                                │ vocab        │
                                │ (id ↔ bytes) │
                                └──────────────┘
```

- **Tokens are handled as bytes.** Some tokens are only part of a multi-byte
  character and are not valid text alone, so the vocabulary maps ids to raw bytes.
- **Masks are precomputed.** Scanning the whole vocabulary at every step would be
  wasteful, so `SharedMasks` builds the masks once at startup.
- **Constraints keep no state.** Everything they need is in `buf`, so one instance
  is reused for every prompt and nothing carries over between prompts.
- **Two short prompts instead of one long one.** The function is chosen from a
  prompt listing every function. Arguments are extracted from a prompt listing
  only the chosen one.
- **The request is embedded in the JSON scaffold.** This places it right before
  the point where the model chooses a name and copies values.

## Performance Analysis

On the 11 public prompts, the program takes 1 min 15 s, within the 5-minute limit.
Every output is valid and every function is chosen correctly, but only 9 of the 11
calls are fully correct, **below the 90% accuracy target**.

### Speed

Nearly all the run time is spent in model calls, and each call reprocesses the
whole sequence. Speed therefore depends on making fewer and shorter calls. Each
row adds one change:

| Version | Time |
|---|---|
| Initial version | about 8 min |
| Skip forced tokens in function names | 6 min 50 s |
| Separate prompts for name and arguments | 6 min 50 s |
| Treat closing tokens as one group | **1 min 15 s** |

Separate prompts sped up the simple requests, but the total did not change because
regex requests got slower: their strings often ran until the length cap. Grouping
the closing tokens let those strings end where the model wanted, which is where
most of the time was saved.

### Accuracy and reliability

Each function was called with the generated parameters, and its result was
compared with the expected one:

| Measure | Result |
|---|---|
| Valid JSON with the expected keys and types | 11/11 |
| Function chosen correctly | 11/11 |
| Fully correct | 9/11 (81.8%) |

Validity comes from the masking, so it does not depend on the model behaving well.
Accuracy does, and both failures are in the values of regex requests:

| Prompt | Generated | Expected |
|---|---|---|
| Replace all numbers in "Hello 34 I'm 233 years old" with NUMBERS | regex `34`, replacement `34` | regex `\d+`, replacement `NUMBERS` |
| Replace all vowels in 'Programming is fun' with asterisks | replacement ` *` | replacement `*` |


## Challenges Faced

- **No KV cache:** private SDK methods and direct PyTorch use are forbidden, so
  every call reprocesses the whole sequence. Time had to be saved by making fewer
  and shorter calls.
- **Runaway strings:** with only the bare `"` allowed, the model's preferred
  closing tokens were forbidden. At the end of one regex, `",` ranked first and the
  bare `"` ranked 157th, so the model kept writing. Each extra token also made
  later calls slower.
- **Tradeoff in shorter prompts:** removing examples improved speed and function
  choice, but exposed the runaway strings until the closing tokens were handled.
- **Numbers and strings end differently:** the model writes the `,` or `}` that ends
  a number, but the program writes what follows a string, so the two need different
  closing text.
- **Exact extraction from a small model:** free-form values such as regexes remain
  the hardest part, and this is still unresolved.

## Testing Strategy

The tests use `pytest` and run with `make test`. They do not need the model to be
downloaded, except for one tokenizer check that is skipped if it is unavailable.

- **`test_loader.py`:** valid input, missing files and invalid data.
- **`test_vocab.py`:** decoding the byte-level vocabulary.
- **`test_constraint.py`:** the rules for each value type.
- **`test_context.py`:** the exact text of the prompts, and their size limit.
- **`test_decoder.py`:** the generation loop, using three fake models:
    - a **teacher** that wants one exact output, which must be reproduced exactly;
    - a **random** model, whose output must still always be valid;
    - a **stubborn** model that never stops, which only the length cap can end.

The code is also checked with `flake8` and `mypy` through `make lint`.

## Example Usage

```sh
uv run python -m src --output data/output/function_calling_results.json
```

```text
[1/11] 'What is the sum of 2 and 3?'
    -> fn_add_numbers  ['a', 'b']
[2/11] 'What is the sum of 265 and 345?'
    -> fn_add_numbers  ['a', 'b']
...
wrote 11/11 result(s) to data/output/function_calling_results.json
```

The output file contains one object per prompt:

```json
[
    {
        "prompt": "Greet shrek",
        "name": "fn_greet",
        "parameters": {
            "name": "shrek"
        }
    },
    {
        "prompt": "What is the square root of 16?",
        "name": "fn_get_square_root",
        "parameters": {
            "a": 16.0
        }
    }
]
```

## Resources

- [JSON Schema Types](https://json-schema.org/understanding-json-schema/reference/type)
- [Qwen Function Calling Format](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [GPT-2 byte-to-unicode encoding](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
- [Pydantic documentation](https://docs.pydantic.dev)
- [NumPy documentation](https://numpy.org/doc)

### AI Usage

AI was used for the following tasks:

- **Performance:** measuring where the run time went, and proposing the name token
  skip, the prompt split and the grouping of closing tokens.
- **Comparison:** reading peer repositories and summarizing what could be useful.
- **Tests:** proposing and generating tests based on the given contract of each file.
- **Evaluation:** checking the generated results and explaining the failures.
- **Documentation:** drafting this README.

Each change was explained and reviewed before being applied.
