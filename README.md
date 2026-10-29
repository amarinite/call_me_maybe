*This project has been created as part of the 42 curriculum by amarquez*

# Call Me Maybe

## Description

A function calling system

## Instructions

### Installation

Clone the repo and blabla

### Running

Run stuff

### Other commands

## Algorithm Explanation

## Design Decisions

```text
┌──────────────────────────────────────────────┐
│  __main__.py                                 │
│  Orchestration: read files, loop prompts,    │
│  write results.                              │
└──────────────────────────────────────────────┘
                    │
    ┌───────────────┼────────────────┐
    ▼               ▼                ▼
┌────────┐   ┌──────────────┐   ┌──────────────┐
│ loader │   │ context      │   │ constraint   │
│        │   │              │   │              │
│ file → │   │ schema +     │   │ generated    │
│ model  │   │ prompt → LLM │   │ text → legal │
│        │   │ input text   │   │ token IDs    │
└────────┘   └──────────────┘   └──────────────┘
                                       │
                                       ▼
                                ┌──────────────┐
                                │ vocab        │
                                │ (id ↔ string)│
                                └──────────────┘
```

## Performance Analysis

## Challenges Faced

## Testing Strategy

## Example Usage

## Resources
[JSON Schema Types](https://json-schema.org/understanding-json-schema/reference/type)
[Qwen Function Calling Format](https://qwen.readthedocs.io/en/latest/framework/function_call.html)

### AI Usage
