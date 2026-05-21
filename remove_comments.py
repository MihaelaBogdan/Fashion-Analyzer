import re
import tokenize
import io

def remove_comments(source):
    result = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        return source

    prev_end_row = 1
    prev_end_col = 0
    output_lines = source.splitlines(keepends=True)
    
    lines_to_remove = set()
    inline_to_trim = {}

    for tok_type, tok_string, tok_start, tok_end, tok_line in tokens:
        if tok_type == tokenize.COMMENT:
            row, col = tok_start
            if col == 0 or tok_line[:col].strip() == '':
                lines_to_remove.add(row)
            else:
                inline_to_trim[row] = col

    new_lines = []
    for i, line in enumerate(output_lines, start=1):
        if i in lines_to_remove:
            continue
        elif i in inline_to_trim:
            col = inline_to_trim[i]
            new_lines.append(line[:col].rstrip() + '\n')
        else:
            new_lines.append(line)

    result = ''.join(new_lines)
    
    # Collapse multiple consecutive blank lines into one
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result

files = ['app.py', 'setup_demo.py', 'setup_clip.py', 'patch_app.py', 'tab3.py']
for fname in files:
    try:
        with open(fname, encoding='utf-8') as f:
            original = f.read()
        cleaned = remove_comments(original)
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        print(f'Cleaned: {fname}')
    except Exception as e:
        print(f'Skipped {fname}: {e}')
