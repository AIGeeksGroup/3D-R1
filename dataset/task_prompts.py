# -*- coding: utf-8 -*-

"""
Task prompts for different 3D-R1 tasks
This file contains the prompt templates used for various tasks including QA, dense captioning, dialogue, and planning.
"""

TASK_PROPMT = {
    'densecap': [
        dict(
            instruction='### human: given the 3D scene, describe this object. ### assistant:',
            answer='{caption}',
        ),
        dict(
            instruction='### human: describe this object in the given 3D scene. ### assistant:',
            answer='{caption}',
        ),
        dict(
            instruction='### human: given the 3D scene, localize and describe this object. ### assistant:',
            answer='the object is localized at {locations}, {caption}',
        ),
        dict(
            instruction='### human: localize and describe this object in the given 3D scene. ### assistant:',
            answer='the object is localized at {locations}, {caption}',
        ),
        dict(
            instruction='### human: given the 3D scene, describe this object first, then localize it. ### assistant:',
            answer='{caption}. It is localized at {locations}',
        ),
        dict(
            instruction='### human: describe then localize the object from the 3D scene. ### assistant:',
            answer='{caption}. It is localized at {locations}',
        ),
    ],
    'ov-det': [
        dict(
            instruction='### human: what is this object? ### assistant:',
            answer='the {category} is localized at {locations}, {caption}',
        ),
    ],
    'qa': [
        dict(
            instruction='### human: given the 3D scene, answer the question: "{question}" ### assistant:',
            answer='{answer}',
            do_localize=False
        ),
        dict(
            instruction='### human: answer this quesiton according to the given 3D scene: "{question}" ### assistant:',
            answer='{answer}',
            do_localize=False
        ),
        dict(
            instruction='### human: answer the question: "{question}" with the related object locations in the input 3D scene. ### assistant:',
            answer='the answer is: {answer}, and the related objects are localized at {locations}',
            do_localize=True
        ),
        dict(
            instruction='### human: given the 3D scene, localize all the related objects first, then answer the question: "{question}" ### assistant:',
            answer='the related objects are localized at {locations}, the answer is: {answer}',
            do_localize=True
        ),
    ],
    'chat': [
        dict(
            instruction='### human: {question} ### assistant:',
            answer='{answer}',
        ),
        dict(
            instruction='### human: based on the 3D scene, {question} ### assistant:',
            answer='{answer}',
        ),
        dict(
            instruction='### human: looking at this 3D environment, {question} ### assistant:',
            answer='{answer}',
        ),
        dict(
            instruction='### human: in this 3D scene, {question} ### assistant:',
            answer='{answer}',
        ),
    ],
    'planning': [
        dict(
            instruction='### human: given this 3D scene, create a plan to {question} ### assistant:',
            answer='{answer}',
        ),
        dict(
            instruction='### human: based on the 3D environment, plan how to {question} ### assistant:',
            answer='{answer}',
        ),
        dict(
            instruction='### human: looking at this 3D scene, what is the best way to {question} ### assistant:',
            answer='{answer}',
        ),
        dict(
            instruction='### human: in this 3D environment, develop a strategy to {question} ### assistant:',
            answer='{answer}',
        ),
    ],
}

# Format strings for different types of annotations
BOX_FORMAT = '<obj>{}, {}, {}, {}, {}, {}</obj>'
COORD_FORMAT = '<loc>{}, {}</loc>'

# Additional format strings for different coordinate systems
POINT_FORMAT = '<point>{}, {}, {}</point>'
REGION_FORMAT = '<region>{}, {}, {}, {}, {}, {}</region>'

# Task-specific format strings
ANSWER_FORMAT = '<answer>{}</answer>'
THINK_FORMAT = '<think>{}</think>'
DESCRIPTION_FORMAT = '<desc>{}</desc>'
LOCATION_FORMAT = '<loc>{}</loc>'

# Special tokens for different tasks
SPECIAL_TOKENS = {
    'qa': ['<answer>', '</answer>', '<think>', '</think>'],
    'densecap': ['<desc>', '</desc>', '<loc>', '</loc>'],
    'chat': ['<response>', '</response>'],
    'planning': ['<plan>', '</plan>', '<step>', '</step>'],
}

# Task descriptions for reference
TASK_DESCRIPTIONS = {
    'qa': 'Question Answering in 3D scenes',
    'densecap': 'Dense Captioning of 3D objects',
    'ov-det': 'Open Vocabulary Object Detection',
    'chat': 'Dialogue about 3D scenes',
    'planning': 'Embodied Planning in 3D environments',
}

# Default prompt selection strategies
PROMPT_SELECTION_STRATEGIES = {
    'qa': 'random',  # Random selection from available prompts
    'densecap': 'random',
    'ov-det': 'first',  # Always use first prompt
    'chat': 'random',
    'planning': 'random',
}

def get_task_prompts(task_name: str):
    """
    Get prompts for a specific task
    
    Args:
        task_name: Name of the task ('qa', 'densecap', 'chat', 'planning', etc.)
        
    Returns:
        List of prompt dictionaries for the task
    """
    return TASK_PROPMT.get(task_name, [])

def get_special_tokens(task_name: str):
    """
    Get special tokens for a specific task
    
    Args:
        task_name: Name of the task
        
    Returns:
        List of special tokens for the task
    """
    return SPECIAL_TOKENS.get(task_name, [])

def format_box_coordinates(coords: list) -> str:
    """
    Format 3D bounding box coordinates using BOX_FORMAT
    
    Args:
        coords: List of 6 coordinates [cx, cy, cz, w, h, l]
        
    Returns:
        Formatted string with box coordinates
    """
    if len(coords) != 6:
        raise ValueError(f"Expected 6 coordinates, got {len(coords)}")
    return BOX_FORMAT.format(*coords)

def format_point_coordinates(coords: list) -> str:
    """
    Format 3D point coordinates using POINT_FORMAT
    
    Args:
        coords: List of 3 coordinates [x, y, z]
        
    Returns:
        Formatted string with point coordinates
    """
    if len(coords) != 3:
        raise ValueError(f"Expected 3 coordinates, got {len(coords)}")
    return POINT_FORMAT.format(*coords)

def format_answer(text: str) -> str:
    """
    Format answer text using ANSWER_FORMAT
    
    Args:
        text: Answer text
        
    Returns:
        Formatted answer string
    """
    return ANSWER_FORMAT.format(text)

def format_think(text: str) -> str:
    """
    Format thinking text using THINK_FORMAT
    
    Args:
        text: Thinking text
        
    Returns:
        Formatted thinking string
    """
    return THINK_FORMAT.format(text)

