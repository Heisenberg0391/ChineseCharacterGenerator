# coding=utf-8
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import config
import os
import glob
import cv2
import mahotas
import progressbar

def augmentation(img, mode, size):
    ''' 不能直接在原始image上改动
        添加随机模糊和噪声
    '''
    image = img.copy()
    # 高斯模糊
    if mode == 0:
        image = cv2.GaussianBlur(image,(5, 5), np.random.randint(1, 10))

    # 模糊后二值化，虚化边缘
    if mode == 1:
        image = cv2.GaussianBlur(image, (5, 5), np.random.randint(1, 6))
        T = mahotas.thresholding.otsu(image)
        thresh = image.copy()
        thresh[thresh > T] = 255
        thresh[thresh < 255] = 0
        image = thresh

    # 横线干扰
    if mode == 2:
        for i in range(0, image.shape[0], 2):
            cv2.line(image, (0, i), (size[0], i), 0, 1)

    # 竖线
    if mode == 3:
        for i in range(0, image.shape[1], 2):
            cv2.line(image, (i, 0), (i, size[0]), 0, 1)

    # 十字线
    if mode == 4:
        for i in range(0,image.shape[0], 2):
            cv2.line(image, (0, i), (size[0], i), 0, 1)
        for i in range(0, image.shape[0], 2):
            cv2.line(image, (i, 0), (i, size[0]), 0, 1)

    # 左右运动模糊
    if mode == 5:
        kernel_size = 7
        kernel_motion_blur = np.zeros((kernel_size, kernel_size))
        kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel_motion_blur = kernel_motion_blur / kernel_size
        image = cv2.filter2D(image, -1, kernel_motion_blur)

    # 上下运动模糊
    if mode == 6:
        kernel_size = 9
        kernel_motion_blur = np.zeros((kernel_size, kernel_size))
        kernel_motion_blur[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
        kernel_motion_blur = kernel_motion_blur / kernel_size
        image = cv2.filter2D(image, -1, kernel_motion_blur)

    # 高斯噪声
    if mode == 7:
        row, col = image.shape
        mean = 0
        sigma = 2
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss
        image = noisy.astype(np.uint8)

    return image

# 根据字体输出图像
def draw_txt(n, charset, fonts, size):
    img_w, img_h = (size[0], size[1])
    factor = 1  # 初始字体大小
    # 初始化进度条
    widgets = ["数据集创建中: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=n,
                                   widgets=widgets).start()
    # 遍历所有字
    for i in range(n):
        char = charset[i]  # 当前字
        # 遍历字体
        for j, each in zip(range(len(fonts)), fonts):
            # 数据增强
            for mode in range(0, 8):
                # 创建画布
                canvas = np.zeros(shape=(img_w, img_h), dtype=np.uint8)
                canvas[0:] = 255
                # 从ndarray转成image进行渲染
                ndimg = Image.fromarray(canvas).convert('RGBA')
                draw = ImageDraw.Draw(ndimg)

                font = ImageFont.truetype(each, int(img_h * factor), 0)
                text_size = font.getsize(char)  # 获取当前字体下的文本区域大小

                # 自动调整字体大小避免超出边界, 至少留白水平10%
                margin = [img_w - int(0.2 * img_w), img_h - int(0.2 * img_h)]
                while (text_size[0] > margin[0]) or (text_size[1] > margin[1]):
                    factor -= 0.01  # 控制字体大小
                    font = ImageFont.truetype(each, int(img_h * factor), 0)  # 加载字体
                    text_size = font.getsize(char)

                # 随机平移
                horizontal_space = int(img_w - text_size[0])
                vertical_space = int(img_h - text_size[1])
                start_x = np.random.randint(1, horizontal_space - 1)
                start_y = np.random.randint(1, vertical_space - 1)

                # 绘制当前文本行
                draw.text((start_x, start_y), char, font=font, fill=(0, 0, 0, 255))
                img_array = np.array(ndimg)
                # ndimg.show()
                # 转灰度图
                img = img_array[:, :, 0]  # [32, 256, 4]
                # 生成保存路径
                save_path = os.path.join(config.IMAGE_PATH, charset[i])
                img_name = save_path + '/' + str(i) + '_' + str(j) + '_' + str(mode) + '.jpg'
                # 数据增强
                aug = augmentation(img, mode, size)
                out = Image.fromarray(aug)

                # 检查路径是否存在，如果存在则直接保存图像
                # 否则需先创建路径
                if os.path.isdir(save_path):
                    out.save(img_name)
                else:
                    os.makedirs(save_path)
                    out.save(img_name)

        pbar.update(i)
    pbar.finish()

# 自动加载字体文件
def load_fonts():
    fnts = []

    # 字体路径
    font_path = os.path.join(config.FONT_PATH, "*.ttf")
    # 获取全部字体路径，存成list
    fonts = list(glob.glob(font_path))

    # 遍历字体文件
    for each in fonts:
        fnts.append(each)

    return fnts


if __name__ == '__main__':
    # 定义一些数组和参数
    label = []
    training_data = []
    # 批大小
    batchSize = 1024
    # 图像尺寸
    size = (64, 64)  # w, h

    # 字体list，每一个字符遍历所有字体，依次输出
    factor = 1
    fonts = load_fonts()

    # 字符集，将其中的字符保存成图像
    charset = u"的一是不人有了在你我个大中要这为上生时会以就子到来" \
              u"可能和自们年多发心好用家出关长他成天对也小后下学都" \
              u"点国过地行信方得最说二业分作如看女于面注别经动公开" \
              u"现而美么还事己理维没之情高法全很日体里工微者实力做" \
              u"等水加定果去所新活着让起市身间码品进孩前想道种识按" \
              u"同车本然月机性与那无手爱样因老内部每更意号电其重化" \
              u"当只文入产合些她三费通但感常明给主名保提将元话气从" \
              u"教相平物场量资知或外度金正次期问放头位安比真务男第" \
              u"解原制区消路及色网花把打吃系回此应友选什表商再万妈" \
              u"被并两题服少风食变容员交儿质建民价养房门需影请利管" \
              u"白简司代口受图处才特报城单西完使已目收十候山数展快" \
              u"强式精结东师求接至海片清各直带程世向先任记持格总运" \
              u"联计觉何太线又免热件权调专医乐效神击设钱健流由见台" \
              u"几增病投易南导功介证走今光朋即视造您立改母推眼复政" \
              u"买传认非基宝营院四习越包游转技条息血科难规众喜便创" \
              u"干界示广红住欢源指该观读享深油达告具取轻康型周装张" \
              u"五满店亲标查育配字类优始整据考案北它客火必购办社命" \
              u"味步护术阅吧素户往菜适边却失节料较形近级准皮衣书马" \
              u"超照值父怎试空切找华供米企助反望香足福且排阳统未治" \
              u"决确项除低根岁则百备像早领酒款防集环富财跟致瘦速择" \
              u"温销团离呢议论吗王州态思参许远责布编随细春克听减言" \
              u"招组景穿黄药肉售股首限检修验共约段笑洗况续底园帮引" \
              u"婚份历济险士错语村伤局票善校战际益职够晚极支存旅故" \
              u"含算送诉留角松积省仅江境称半星升象材预群获青终害肤" \
              u"属显卡餐银声站队落假县饭补研连德哪钟遇黑双待毒断充" \
              u"智演讲压农愿尽拉粉响死牌古货玩苦率千施蛋器楼痛究睡" \
              u"状订义绝石亮势音搭委斯居李紧坚脸独依丽严止疗右喝鸡" \
              u"牛林板某负京丰句评融军懂吸划念夫层降哦税豆彩官络胸" \
              u"拿画尔龙察班构秘否叫球幸座慢兴佛室啊均付模协互置般" \
              u"英净换短左版课茶策毛停河肥答良久承控激范章云普套另" \
              u"奖须例写灵担志顾草镇退希谢爸采六鱼围密庭脑奇八卖童" \
              u"土圈谁拥糖监甚怕贵顺鲜冷差梦警拍铁亿争夜背永街律饮" \
              u"继刻初突倒聘木熟婆列频虽刚妆举尚汽曾脚奶破静驾块蓝" \
              u"酸核锅艺绿博额陈坐靠巧掉飞盘币腿巴培若闻史亚纸症季" \
              u"叶乡丝询剧礼七址添织略虚迎摄余乎缺胃爆域妻练荐临佳" \
              u"府追患树颜诚伴湖贴午困似测肝归宁暖纳宜阿异卫录液私" \
              u"谈泡惊索盐漂损稳休折讯堂怀惠汤纪散藏湿透令冰妇麻醒" \
              u"宣抗典执秀肌训刘急赶播苏淡革阴批盖腰肠脱印硬促冲床" \
              u"努脏跑雅厅罗惯族姐犯罪赛趣骨烧哈避征劳载润炒软慧驶" \
              u"妹占租馆累签副键煮尊予缘港雨兰斤呼申障坏竟疑顶饰九" \
              u"炎歌审戏借误辆端沙掌恶疾露括固移脂武寒零烟毕雪登朝" \
              u"聚笔姓波涨救厂央咨党延耳危斑汉沉夏侧鞋牙媒腹龄励瓜" \
              u"敢忙宽箱释操输抱野癌守搞染姜默翻哥洁娘挑凉末潮违附" \
              u"杀宫迷杂弱岛础贫析乱乳辣弃桃轮浪赏抽镜盛胜玉烦植绍" \
              u"恋冒缓渐虑肯赚绩忘珍恩针猪既聊蜜握舞甜败汇抓刺骗杯" \
              u"啦灯赞寻仍陪涉椒荣哭欲词巨圆刷概沟幼尤偏斗胡启尼述" \
              u"弟屋田判触柔忍架吉肾狗欧遍甘瓶综曲威齐桥纯阶贷丁伙" \
              u"眠罚逐韩封扎厚著督冬舒杨惜汁庆迪洋洲旧映疼席暴漫辈" \
              u"射鼓葱侵羊倍挂束幅碗裤胖旺川搜航弹嘴派脾届托库唯奥" \
              u"菌君途讨券距粗诗授祛谓序账凡晓峰剂筑敏肚暗辑访岗腐" \
              u"痘摩烈扬谷纹遗偿穷帝尿腾禁竞豪苹跳挥抢卷胆递珠敬甲" \
              u"乘孕绪纷隐滑浓膜姑探宗姻诺摆狂篇睛闲勇蒜尾旦庄窗扫" \
              u"辛陆塑幕聪详污圳扮肿楚忆匀炼耐衡措铺薪泰懒贝磨怨鼻" \
              u"圣孙眉泉洞焦毫戴旁符泪邮爷钢混厨抵灰献扣怪碎擦胎缩" \
              u"扶恐欣顿伟丈皇蒙胞尝寿攻仁津潜滴晨颗舍秒刀酱悲妙隔" \
              u"桌册迹仔闭奋袋墙嫌萝唐跌尖莫拌赔忽宿扩胶雷燕衰挺宋" \
              u"湾脉凭丹繁拒肺涂郁剩仪紫滋泽薄森唱残虎档猫麦劲偶秋" \
              u"疯俗悉弄船雄兵晒扰蒸悟肪览籍丑拼诊吴循偷灭伸赢魅勤" \
              u"旗亡乏估替吐碰淘彻逼氧梅遭孔稿嘉卜赵姿储呈乌娱闹裙" \
              u"倾震忧貌萨塞鬼池沿畅盟仙醋炸粥咖瑜返稍灾肩殊逃荷描" \
              u"朱朵横徐杰陷迟莱纠榜债烂伽拟匙圾巾恼誉垃颈壁链糊悦" \
              u"屏浮魔毁拜宾迅芝燃迫疫柜烤塔赠伪阻绕饱辅醉抑撒粘丢" \
              u"卧徒奔锁董枣截番蔬摇亦趋冠呀疲婴诸贸泥伦嫁祖朗琴拔" \
              u"孤摸壮帅阵梁宅啥伊鲁怒熊艾裁犹撑莲吹纤昨谱咳蜂闪嫩" \
              u"瞬霸兼恨昌踏瑞樱萌厕郑寺愈傻慈汗奉缴暂爽堆浙忌慎坦" \
              u"撞耗粒仿诱凤矿锻馨尘兄杭虫熬赖恰恒鸟猛唇幻窍浸诀填" \
              u"亏覆盆彼腺胀苗竹魂吵刑惑岸傲垂呵荒页抹揭贪宇泛劣臭" \
              u"呆梯啡径咱筹娃鉴禅召艳澳恢践迁废燥裂兔溪梨饼勺碍穴" \
              u"坛诈宏井仓删挣柳戒腔涵寸弯隆插祝氏泌盒邀煤膏棒跨拖" \
              u"葡骂喷肖洛盈浅逆夹贤晶厌侠欺敌钙冻盾桂仰滚萄厦牵疏" \
              u"齿挡孝滨吨渠囊慕捷淋桶脆沫辉耀渴邪轨悔猎煎沈虾醇贯" \
              u"衫荡谋携晋糕玻肃杜皆秦盗臂舌杆俱棉挤拨剪阔稀腻骑玛" \
              u"忠伯伍狠宠勒浴勿媳晕佩屈纵奈抬栏菲坑茄雾坡幽跃坊枝" \
              u"凝拳谨筋菇锋璃郭钻酷愁摘捐谐遵苍飘搅漏泄祥锦衬矛猜" \
              u"凌挖喊猴芳曼痕鼠允叔牢绘嘛吓振墨烫厉昆拓卵凯淀皱枪" \
              u"尺疆姆笋粮邻菩署柠遮艰芽爬夸捞叹缝妨奏岩寄吊狮剑驻" \
              u"洪夺募凶辨崇莓斜檬悬瘤欠刊曝傅悠椅戳棋慰丧拆绵炉徽" \
              u"驱曹履俄兑闷赋狼愉纽膝饿窝辞躺瓦逢堪薯哟袭壳咽岭槽" \
              u"雕昂闺御旋寨抛祸殖喂俩贡狐弥遥桑搬陌陶乃寂滩妥辰堵" \
              u"蛇侣邦蝙陵洒浆蹲惧霜丸娜扔肢姨援炫岳迈躁蝠埋泻巡溶" \
              u"氛械翠陕乔漠滞哲浩驰摊糟铜赤谅蕉昏劝杞扭骤杏娇渡抚" \
              u"羡娶串碧叉廉膀柱垫伏痒捕咸瓣庙敷卑碑沸鸭纱赴盲辽疮" \
              u"浦逛愤黎耍咬仲枸催喉瑰勾臀泼椎翼奢郎杠碳谎悄瓷绑酬" \
              u"菠朴割砖惨埃霍耶仇嗽塘邓漆蹈鹰披橘薇溃拾押炖霉痰袖" \
              u"巢帽宴卓踪屁刮晰豫玫驭羞讼茫厘扑亩鸣罐澡惩刹啤揉纲" \
              u"腥沾陀蟹枕逸橙梳浑毅吕泳碱缠柿砂羽黏芹馈柴侦卢憾疹" \
              u"贺牧俊峡硕倡蓄赌吞躲旨崩寞碌堡唤韭趁惹衷蛮译彭掩攀" \
              u"慌牲栋鼎鹅弘敲诞撕卦腌葛舟寓氨弗伞罩芒沃棚契巷磁浇" \
              u"逻廊棵溢箭匹矩颇爹玲绒雀鸿贩锐曰蕾竭剥沪畏渣歉摔旬" \
              u"颖茂擅铃淮叠挫逗晴柏舰翁框涌琳罢辩勃霞裹烹庸臣莉匆" \
              u"熙轩骄煲掘搓乙痴恭韵渗薏炭痣锡脊夕丘苑蔡裸灌庞龟" \
              u"窄兽敦辟牺僧怜湘篮妖喘瘾蓬挽芦谦踩辱辖捧坠滤炮撩狱" \
              u"亭虹吻煌谊枯脐螺扇抖戚怖帐盼冯劫墓崔酵殿蝶袁袜枚芯" \
              u"绳颠耕壶叨乖呕筷捡鹿潘笨扁渔株斥砸涩倦沥丛翔吼裕翰" \
              u"蒂尸莴暑肴凰馅阁誓匠侯韧钥哒狸媚壤驴逝渍嘲颁谜翅笼" \
              u"冈蓉脖甩扯宙叛帖萧芬潭涛闯泊宰梗鑫祭嚼卸尬尴怡咒晾" \
              u"嚣哄掏哀盯腊灿涯钞轰髦斌茅骚咋茨蝇枢捣顽彰拘坎役砍" \
              u"皂汪孟筱愚滥妒塌轿窃喻胁钓墅糙浏愧赫捏妮溜谣膳郊睫" \
              u"沧撤搏汰鹏菊帘秤衔捉鹤贿廷撼钾绽轴凸魄晃磷蒋栽荆蠢" \
              u"魏蜡缸筒遂茎芭伐邵瞎帕凑唠祈赁秩辫玄酶潇稻兜婷栓屡" \
              u"削钉拭蕴糯煞坪兹妃兆沂纺酿柚瀑稠腕勉疡贱冀跪凹辜铭" \
              u"赐绎灶弛嫉姚慨褐翘饶焯蒲哎僵隙犬剖昧湛矮舆吾剔甄逾" \
              u"虐粹牡莎罕蠕拐琪瑟霾辐帆拇榨冤绣痔筛雇祷歪贼肛垢抄" \
              u"饺琐裔黛睁捂萎酥饥衍靓榄嗯肆咯槛寡诵贬瞧乞贾弓珊眸" \
              u"屑熏籽乾聆狭韦锈毯蹄涤磊赘歇坝豹橄葬竖奴磅蝴淑柯敛" \
              u"侈叙惫俞翡叮蜀逊葩拯咪喔灸橱函厢瑶橡俯沛嘱佣陇莞妄" \
              u"榆淫靖俏敞嫂烘腑崖扒洽宵膨亨妞硫剁秉淤婉稣筝屌挨儒" \
              u"哑铅斩阱钩睿彬啪琼桩萍蔓焖踢铝仗荠棍棕铸榴惕巩杉芋" \
              u"攒髓拦蝎飙栗畜挪冥藤坤嘿磕椰憋荟坞屯饲懈梭夷嘘沐蔗" \
              u"蚕粕吁卉昭饪钮恳睦讶穆拣傍岂蘸噪戈靴瑕龈讽泣浊哇趾" \
              u"蔽丫歧蚊暨钠芪艇暮擎畔禽拧惟俭蔚恤蚀尹侍馒锌骼咏堕" \
              u"渊桐窒焕阀藕耻躯薛菱谭豁昕喧藉丙鸦驼拢奸爪睹绸暧佐" \
              u"颊澜禄缀煸趟揽蘑瘀阜拎屎颤邑胰肇哺噢矫讳雌怠楂苛暇" \
              u"酪佑妍婿耿妊萃灼澄撰弊挚庐雯靡牟硝酮醛苓紊肘趴廓" \
              u"卤昔鄂哮赣汕貅渝媛貔彦荫觅蹭巅岚甸漓迦邂稚濮陋逅窑" \
              u"笈弧颐禾瘙脓刃愣拴旭蚁滔仕荔琢澈睐隶粤盏遣汾镁硅枫" \
              u"淹仆胺娠舅弦殷惰麟苔芙堤旱蛙驳羯涕侨铲糜烯扛腮猿烛" \
              u"昵韶莹洱诠襄棠鸽仑峻啃瞒喇绊胱咙踝褶娩鲍掀漱绅奠芡" \
              u"蜗疤兮矣熔俺掰拱骏贞姥哼倘栖屉眷渭幢芜溺茯袍淳沦绞" \
              u"倪缚碟雁孵粪崛舱褪诡悍芸宪壹诫窟葵呐锤摧碾鞭嗓呱芥"

    # 生成n个字
    n = 10
    draw_txt(n, charset, fonts, size)
