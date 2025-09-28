import re

def compact_image_pads(text: str) -> str:
    compacted_text = re.sub(r'(<\|image_pad\|>)+', r'<|image_pad|>', text)
    return compacted_text

def print_answer(text: str, ground_truth: str):  
    print("predict_str:",compact_image_pads(text))
    print("ground_truth:", ground_truth)  
    
def compute_score(predict_str: str, ground_truth: str, extra_info=None):
    a,b,c,d= v11(predict_str, ground_truth, extra_info)
    print("========================start of text========================")
    print_answer(predict_str,ground_truth)
    print("scores:", a,b,c,d)
    print("========================end of text========================")
    return a,b,c,d

def v11(predict_str: str, ground_truth: str, extra_info=None):
    format_score = 0.0
    acc_score = 0.0
    other_score = 0.0
    total_score = 0.0
    nframes = 8
    question=extra_info['question']
    time_reward=False

    try:
        think_contents = re.findall(r'<think>(.*?)</think>', predict_str, re.DOTALL)
        action_contents = re.findall(r'<action>(.*?)</action>', predict_str, re.DOTALL)
        system_responses = re.findall(r'</action>(.*?)<think>', predict_str, re.DOTALL)
    except Exception as e:
        return total_score, acc_score, format_score, other_score

    if not think_contents or not action_contents or len(think_contents) != len(action_contents):
        return total_score, acc_score, format_score, other_score

    last_action = action_contents[-1].strip()
    answer_match = re.match(r'output answer:\s*(\S+)', last_action)
    if not answer_match:
        return total_score, acc_score, format_score, other_score
    
    action_frame_pairs = [(0,extra_info['total_frames']-1)]
    requested_times = []
    expected_frame_in_next_action = None

    for i, action in enumerate(action_contents[:-1]):
        action = action.strip()
        think_block = think_contents[i]

        if expected_frame_in_next_action is not None:
            frame_match_for_check = re.match(r'choose frames between (\d+) and (\d+)', action)
            if frame_match_for_check:
                start_f = int(frame_match_for_check.group(1))
                end_f = int(frame_match_for_check.group(2))
                if not (start_f <= expected_frame_in_next_action <= end_f):
                    return total_score, acc_score, format_score, other_score
            expected_frame_in_next_action = None

        frame_match = re.match(r'choose frames between (\d+) and (\d+)', action)
        if frame_match:
            num1 = int(frame_match.group(1))
            num2 = int(frame_match.group(2))
            current_pair = (num1, num2)
            check_pair=(num1+1,num2-1)
            if current_pair in action_frame_pairs: 
                return total_score, acc_score, format_score, other_score
            action_frame_pairs.append(current_pair)
            action_frame_pairs.append(check_pair)

            if num1 >= num2 - nframes: 
                return total_score, acc_score, format_score, other_score
            
            numbers_in_think = re.findall(r'(?<!:)\b(\d+)\b(?!:)', think_block)
            if len(numbers_in_think) >= 2:
                num_a, num_b = map(int, numbers_in_think[-2:])
                if (num_a, num_b) != current_pair:
                    return total_score, acc_score, format_score, other_score
            else:
                return total_score, acc_score, format_score, other_score
            continue

        time_match = re.match(r'get frame number at time\s+(\d{1,3}:\d{2})', action)
        if time_match:
            time_str_action = time_match.group(1)
            
            if time_str_action in requested_times: 
                return total_score, acc_score, format_score, other_score
            requested_times.append(time_str_action)

            system_response = system_responses[i].strip()
            response_match = re.search(r'is:\s*(\d+)', system_response)
            if response_match:
                expected_frame_in_next_action = int(response_match.group(1))
                if expected_frame_in_next_action > 0:
                    time_reward = True
            else:
                return total_score, acc_score, format_score, other_score
            continue

        return total_score, acc_score, format_score, other_score
    
    if expected_frame_in_next_action is not None:
        return total_score, acc_score, format_score, other_score
    
    format_score = 1.0
    model_answer = answer_match.group(1)

    if model_answer == ground_truth:
        acc_score = 1.0
        if time_reward:
            time_pattern = re.compile(r'\d+:\d+(:\d+)?')
            if re.search(time_pattern, question):
                other_score += 0.5
        if len(action_frame_pairs)>1:
            other_score += 0.02
    total_score = acc_score + other_score

    return total_score, acc_score, format_score, other_score
