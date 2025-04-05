## Dataset and Format

数据集可从[Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N8XJME)上获取（361 MB）。这是一个压缩后的 CSV 文件，包含实验中使用的 1300 万条 Duolingo 学生学习轨迹。

数据集每列如下：

- `p_recall` - 本课/练习中正确回忆出单词/词素的练习比例
- `timestamp` - 当前课程/练习的 UNIX 时间戳
- `delta` - 自上次包含该单词/词素的课程/练习以来的时间（以秒为单位）
- `user_id` - 上课/练习的学生用户 ID（匿名）
- `learning_language` - 正在学习的语言
- `ui_language` - 用户界面语言（可能是学生的母语）**可以用作敏感属性**
- `lexeme_id` - 词位标签（即单词）的系统 ID
- `lexeme_string` - 词位标签（见下文）

研究以下四列：

- `history_seen` - 用户在本课/练习之前看到该单词/词素的总次数
- `history_correct` - 用户在本课/练习之前正确读出单词/词素的总次数
- `session_seen` - 用户在本课/练习中看到该单词/词素的次数
- `session_correct` - 用户在本课/练习中正确使用单词/词汇的次数



关于`lexeme_id` ：

该列包含 Duolingo 在实验中为每节课/练习（数据实例）使用的 "词素标签 "的字符串表示。为方便今后的研究和分析，本版本增加了这一列。在我们最初的实验中只使用了这一列。该字段使用以下格式：`lexeme_string``lexeme_id``lexeme_string`



```
surface-form/lemma<pos>[<modifiers>...]
```

Where refers to the inflected form seen in (or intended for) the exercise, is the uninflected root, is the high-level part of speech, and each of the encodes a morphological component specific to the surface form (tense, gender, person, case, etc.). 

西班牙语中的几个例子:`surface-form``lemma``pos``modifers`

```
bajo/bajo<pr>
blancos/blanco<adj><m><pl>
carta/carta<n><f><sg>
de/de<pr>
diario/diario<n><m><sg>
ellos/prpers<prn><tn><p3><m><pl>
es/ser<vbser><pri><p3><sg>
escribe/escribir<vblex><pri><p3><sg>
escribimos/escribir<vblex><pri><p1><pl>
lee/leer<vblex><pri><p3><sg>
lees/leer<vblex><pri><p2><sg>
leo/leer<vblex><pri><p1><sg>
libro/libro<n><m><sg>
negra/negro<adj><f><sg>
persona/persona<n><f><sg>
por/por<pr>
son/ser<vbser><pri><p3><pl>
soy/ser<vbser><pri><p1><sg>
y/y<cnjcoo>
```

某些标记在书写时，包含通配符成分。

此文件包含用于词素标记的位置和修饰成分的参考资料：

`<*...>``<*sf>``<*numb>``lexeme_reference.txt`

# 补充

[Duolingo Dataverse (harvard.edu)](https://dataverse.harvard.edu/dataverse/duolingo)

## 数据预处理后的结果

`ui_language_binary`：界面使用的语言，二分类，eng和noneng。这个可作为敏感属性研究。

`learning_language_binary`：学习的语言，二分类，eng和noneng

`is_workday	`：是否是工作日

`history_seen` - 用户在本课/练习之前看到该单词/词素的总次数

`history_correct` - 用户在本课/练习之前正确读出单词/词素的总次数

`session_seen` - 用户在本课/练习中看到该单词/词素的次数

`session_correct` - 用户在本课/练习中正确使用单词/词汇的次数