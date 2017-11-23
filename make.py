import glob
import os.path
import re
import subprocess


def main():
    subprocess.call('git pull origin master', shell=True)
    subprocess.call('mkdocs build', shell=True)

    pattern = re.compile(r'((?<=href=")|((?<=src=")))(?P<src>.*?\.((jpg)|(png)|(gif)|(svg)|(bmp)|(css)|(js)))(?=")')

    def repl(x): return 'http://cdn.accu.cc/{0}'.format(x.group('src').strip('./'))
    for entry in glob.glob('**/*.html', recursive=True):
        print(entry)
        with open(entry, 'r+', encoding='utf-8') as f:
            content = f.read()
            content = pattern.sub(repl, content)
            f.seek(0, 0)
            f.write(content)

    with open('./site/baidu_verify_Pem1L7uAVI.html', 'w') as f:
        f.write('Pem1L7uAVI')
    with open('./site/google9b75b4b4147e247b.html', 'w') as f:
        f.write('google-site-verification: google9b75b4b4147e247b.html')

    if not os.path.exists('site/page'):
        subprocess.call('ln -s -f content page', shell=True, cwd='site')


if __name__ == '__main__':
    main()
