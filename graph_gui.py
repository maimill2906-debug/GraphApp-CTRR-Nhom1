import tkinter as tk
import time
from tkinter import messagebox, filedialog
import heapq
from collections import deque

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng trực quan đồ thị")
        self.root.geometry("1200x650")

        self.G = nx.Graph()
        self.is_directed = False
        self.pos = {}

        left_frame = tk.Frame(root, width=320, bg="#f0f0f0", padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        right_frame = tk.Frame(root, bg="white")
        right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        tk.Label(left_frame, text="CÀI ĐẶT ĐỒ THỊ", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=5)

        self.graph_type_var = tk.StringVar(value="Undirected")
        tk.Label(left_frame, text="Loại đồ thị:", bg="#f0f0f0").pack(anchor="w")
        tk.Radiobutton(left_frame, text="Vô hướng (Undirected)", variable=self.graph_type_var,
                       value="Undirected", command=self.change_graph_type, bg="#f0f0f0").pack(anchor="w")
        tk.Radiobutton(left_frame, text="Có hướng (Directed)", variable=self.graph_type_var,
                       value="Directed", command=self.change_graph_type, bg="#f0f0f0").pack(anchor="w")

        tk.Frame(left_frame, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, pady=8)

        tk.Label(left_frame, text="THÊM CẠNH", font=("Arial", 11, "bold"), bg="#f0f0f0").pack(anchor="w")
        input_frame = tk.Frame(left_frame, bg="#f0f0f0")
        input_frame.pack(fill=tk.X, pady=4)

        tk.Label(input_frame, text="Từ:", bg="#f0f0f0").grid(row=0, column=0)
        self.entry_u = tk.Entry(input_frame, width=8)
        self.entry_u.grid(row=0, column=1)

        tk.Label(input_frame, text="Đến:", bg="#f0f0f0").grid(row=0, column=2)
        self.entry_v = tk.Entry(input_frame, width=8)
        self.entry_v.grid(row=0, column=3)

        tk.Label(input_frame, text="Trọng số:", bg="#f0f0f0").grid(row=1, column=0)
        self.entry_w = tk.Entry(input_frame, width=8)
        self.entry_w.grid(row=1, column=1)
        self.entry_w.insert(0, "1")

        tk.Button(left_frame, text="Thêm cạnh", bg="#4CAF50", fg="white",
                  command=self.add_edge).pack(fill=tk.X, pady=2)
        tk.Button(left_frame, text="Xóa đồ thị", bg="#f44336", fg="white",
                  command=self.clear_graph).pack(fill=tk.X, pady=2)
        tk.Button(left_frame, text="Lưu đồ thị", bg="#2196F3", fg="white",
                  command=self.save_graph).pack(fill=tk.X, pady=2)
        tk.Button(left_frame, text="Tải đồ thị", bg="#9C27B0", fg="white",
                  command=self.load_graph).pack(fill=tk.X, pady=2)

        self.info_label = tk.Label(left_frame, text="Số đỉnh: 0 | Số cạnh: 0",
                                   bg="#f0f0f0", font=("Arial", 9, "italic"))
        self.info_label.pack(anchor="w", pady=5)

        tk.Frame(left_frame, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, pady=8)

        tk.Label(left_frame, text="THUẬT TOÁN ĐỒ THỊ",
                 font=("Arial", 13, "bold"), bg="#f0f0f0").pack(anchor="w")

        tk.Label(left_frame, text="Chọn thuật toán:", bg="#f0f0f0").pack(anchor="w")
        self.algorithm_var = tk.StringVar(value="BFS")
        algo_options = [
            "BFS", "DFS", "Dijkstra",
            "Prim MST", "Kruskal MST",
            "Euler (Hierholzer)", "Ford-Fulkerson (Max Flow)",
            "Check 2-way (u <-> v)", "Convert Representation"
        ]
        self.algo_menu = tk.OptionMenu(left_frame, self.algorithm_var, *algo_options)
        self.algo_menu.pack(fill=tk.X)

        param_frame = tk.Frame(left_frame, bg="#f0f0f0")
        param_frame.pack(fill=tk.X, pady=3)
        tk.Label(param_frame, text="Start:", bg="#f0f0f0").grid(row=0, column=0)
        self.entry_start = tk.Entry(param_frame, width=8)
        self.entry_start.grid(row=0, column=1)

        tk.Label(param_frame, text="End:", bg="#f0f0f0").grid(row=0, column=2)
        self.entry_end = tk.Entry(param_frame, width=8)
        self.entry_end.grid(row=0, column=3)

        tk.Label(left_frame, text="Chế độ trực quan:", bg="#f0f0f0").pack(anchor="w")
        self.visual_mode = tk.StringVar(value="Final")
        tk.Radiobutton(left_frame, text="Final", variable=self.visual_mode, value="Final",
                       bg="#f0f0f0").pack(anchor="w")
        tk.Radiobutton(left_frame, text="Step-by-step", variable=self.visual_mode, value="Step",
                       bg="#f0f0f0").pack(anchor="w")

        tk.Button(left_frame, text="Chạy thuật toán",
                  command=self.run_algorithm,
                  bg="#FF9800", fg="white").pack(fill=tk.X, pady=5)

        tk.Button(left_frame, text="Reset màu", command=self.redraw_only,
                  bg="#607D8B", fg="white").pack(fill=tk.X, pady=2)

        tk.Frame(left_frame, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, pady=8)

        tk.Label(left_frame, text="KẾT QUẢ / LOG", font=("Arial", 10, "bold"),
                 bg="#f0f0f0").pack(anchor="w")
        self.result_text = tk.Text(left_frame, height=10, width=35)
        self.result_text.pack(fill=tk.BOTH)

        self.figure, self.ax = plt.subplots(figsize=(7, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.draw_graph()

        #=======================================================================
    # CHUYỂN ĐỔI LOẠI ĐỒ THỊ
    #=======================================================================
    def change_graph_type(self):
        edges = list(self.G.edges(data=True))
        if self.graph_type_var.get() == "Directed":
            self.G = nx.DiGraph()
            self.is_directed = True
        else:
            self.G = nx.Graph()
            self.is_directed = False
        self.G.add_edges_from(edges)
        self.pos = {}
        self.draw_graph()

    #=======================================================================
    # CÁC HÀM ĐỒ THỊ CƠ BẢN
    #=======================================================================
    def add_edge(self):
        u = self.entry_u.get().strip()
        v = self.entry_v.get().strip()
        w_str = self.entry_w.get().strip()

        if not u or not v:
            messagebox.showwarning("Lỗi", "Vui lòng nhập đủ 2 đỉnh!")
            return

        try:
            w = float(w_str)
        except:
            w = 1.0

        self.G.add_edge(u, v, weight=w)
        self.entry_u.delete(0, tk.END)
        self.entry_v.delete(0, tk.END)
        self.pos = {}
        self.draw_graph()

    def clear_graph(self):
        self.G.clear()
        self.pos = {}
        self.draw_graph()
        self.result_text.delete("1.0", tk.END)

    def save_graph(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt")
        if not path: return
        nx.write_edgelist(self.G, path, data=['weight'])
        messagebox.showinfo("OK", "Đã lưu.")

    def load_graph(self):
        path = filedialog.askopenfilename()
        if not path: return
        if self.is_directed:
            self.G = nx.read_edgelist(path, data=(("weight", float),), create_using=nx.DiGraph())
        else:
            self.G = nx.read_edgelist(path, data=(("weight", float),), create_using=nx.Graph())
        self.pos = {}
        self.draw_graph()

    #=======================================================================
    # VẼ ĐỒ THỊ
    #=======================================================================
    def draw_graph(self, highlight_nodes=None, highlight_edges=None, edge_color_map=None):
        self.ax.clear()
        n = self.G.number_of_nodes()
        e = self.G.number_of_edges()

        self.info_label.config(text=f"Số đỉnh: {n} | Số cạnh: {e}")

        if n > 0:
            if not self.pos or set(self.pos.keys()) != set(self.G.nodes()):
                self.pos = nx.spring_layout(self.G, seed=42)

            highlight_nodes = set(highlight_nodes or [])

            node_colors = [
                "orange" if node in highlight_nodes else "skyblue"
                for node in self.G.nodes()
            ]

            edges = list(self.G.edges())
            colors = []
            for u, v in edges:
                if highlight_edges and ((u, v) in highlight_edges or (v, u) in highlight_edges):
                    colors.append("red")
                elif edge_color_map and ((u, v) in edge_color_map or (v, u) in edge_color_map):
                    colors.append(edge_color_map.get((u, v), edge_color_map.get((v, u))))
                else:
                    colors.append("gray")

            nx.draw(
                self.G, self.pos, ax=self.ax,
                node_color=node_colors, node_size=1200,
                edge_color=colors,
                with_labels=True,
                arrows=self.is_directed,
                font_size=10
            )

            lbls = nx.get_edge_attributes(self.G, "weight")
            nx.draw_networkx_edge_labels(self.G, self.pos, ax=self.ax, edge_labels=lbls)

        self.ax.set_title("Đồ thị có hướng" if self.is_directed else "Đồ thị vô hướng")
        self.ax.axis("off")
        self.canvas.draw()

    def redraw_only(self):
        self.draw_graph()

    #=======================================================================
    # CHẠY THUẬT TOÁN
    #=======================================================================
    def run_algorithm(self):
        algo = self.algorithm_var.get()
        start = self.entry_start.get().strip()
        end = self.entry_end.get().strip()

        start = start if start else None
        end = end if end else None

        self.result_text.delete("1.0", tk.END)

        if algo == "BFS":
            self.handle_bfs(start, end)
        elif algo == "DFS":
            self.handle_dfs(start, end)
        elif algo == "Dijkstra":
            self.handle_dijkstra(start, end)
        elif algo == "Prim MST":
            self.handle_prim(start)
        elif algo == "Kruskal MST":
            self.handle_kruskal()
        elif algo == "Euler (Hierholzer)":
            self.handle_euler(start)
        elif algo == "Ford-Fulkerson (Max Flow)":
            self.handle_max_flow(start, end)
        elif algo == "Check 2-way (u <-> v)":
            self.handle_two_way(start, end)
        elif algo == "Convert Representation":
            self.handle_convert_representation()

    #=======================================================================
    # BFS - IN ORDER + PATH
    #=======================================================================
    def handle_bfs(self, start, end):
        if start is None or start not in self.G:
            messagebox.showwarning("Lỗi", "Start không hợp lệ.")
            return

        parent = {start: None}
        visited = []
        q = deque([start])

        # --- FINAL MODE ---
        if self.visual_mode.get() == "Final":
            while q:
                u = q.popleft()
                if u in visited: 
                    continue
                visited.append(u)

                for v in self.G.neighbors(u):
                    if v not in visited and v not in q:
                        parent[v] = u
                        q.append(v)

            self.result_text.insert(tk.END, "BFS order: " + " -> ".join(visited) + "\n")

            if end not in parent:
                self.result_text.insert(tk.END, "Không có đường đi.\n")
                self.draw_graph(highlight_nodes=visited)
                return

            # reconstruct path
            path = []
            x = end
            while x is not None:
                path.append(x)
                x = parent[x]
            path.reverse()

            self.result_text.insert(tk.END, "Path: " + " -> ".join(path) + "\n")

            edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            self.draw_graph(highlight_nodes=path, highlight_edges=edges)
            return

        # --- STEP-BY-STEP MODE ---
        self.result_text.insert(tk.END, "BFS Animation...\n")
        self.draw_graph()

        while q:
            u = q.popleft()
            if u in visited:
                continue
            visited.append(u)

            self.draw_graph(highlight_nodes=[u])
            self.root.update()
            time.sleep(0.5)

            for v in self.G.neighbors(u):
                if v not in visited and v not in q:
                    parent[v] = u
                    q.append(v)

                    self.draw_graph(highlight_nodes=[u, v], highlight_edges=[(u, v)])
                    self.root.update()
                    time.sleep(0.5)

        order = visited
        self.result_text.insert(tk.END, "BFS order: " + " -> ".join(order) + "\n")

        if end not in parent:
            self.result_text.insert(tk.END, "Không có đường đi.\n")
            return

        # reconstruct path
        path = []
        x = end
        while x is not None:
            path.append(x)
            x = parent[x]
        path.reverse()

        self.result_text.insert(tk.END, "Path: " + " -> ".join(path) + "\n")

        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        self.draw_graph(highlight_nodes=path, highlight_edges=edges)

    #=======================================================================
    # DFS
    #=======================================================================
    def handle_dfs(self, start, end):
        if start is None or start not in self.G:
            messagebox.showwarning("Lỗi", "Start không hợp lệ.")
            return

        visited = []
        stack = [start]
        parent = {start: None}

        # --- FINAL MODE ---
        if self.visual_mode.get() == "Final":
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.append(u)

                neighbors = list(self.G.neighbors(u))
                neighbors.reverse()

                for v in neighbors:
                    if v not in visited:
                        parent[v] = u
                        stack.append(v)

            self.result_text.insert(tk.END, "DFS order: " + " -> ".join(visited) + "\n")

            if end not in parent:
                self.draw_graph(highlight_nodes=visited)
                return

            path = []
            x = end
            while x is not None:
                path.append(x)
                x = parent[x]
            path.reverse()

            self.result_text.insert(tk.END, "Path DFS: " + " -> ".join(path) + "\n")

            edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            self.draw_graph(highlight_nodes=path, highlight_edges=edges)
            return

        # --- STEP MODE ---
        self.result_text.insert(tk.END, "DFS Animation...\n")
        self.draw_graph()

        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.append(u)

            self.draw_graph(highlight_nodes=[u])
            self.root.update()
            time.sleep(0.5)

            neighbors = list(self.G.neighbors(u))
            neighbors.reverse()
            for v in neighbors:
                if v not in visited:
                    parent[v] = u
                    stack.append(v)
                    self.draw_graph(highlight_nodes=[u, v], highlight_edges=[(u, v)])
                    self.root.update()
                    time.sleep(0.5)

        self.result_text.insert(tk.END, "DFS order: " + " -> ".join(visited) + "\n")

        if end not in parent:
            return

        path = []
        x = end
        while x is not None:
            path.append(x)
            x = parent[x]
        path.reverse()

        self.result_text.insert(tk.END, "Path DFS: " + " -> ".join(path) + "\n")

        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        self.draw_graph(highlight_nodes=path, highlight_edges=edges)

    #=======================================================================
    # DIJKSTRA
    #=======================================================================
    def handle_dijkstra(self, start, end):
        if start not in self.G or end not in self.G:
            messagebox.showwarning("Lỗi", "Start/End không hợp lệ.")
            return

        dist = {n: float("inf") for n in self.G.nodes()}
        prev = {n: None for n in self.G.nodes()}
        dist[start] = 0
        pq = [(0, start)]

        # --- FINAL MODE ---
        if self.visual_mode.get() == "Final":
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                if u == end:
                    break

                for v in self.G.neighbors(u):
                    w = self.G[u][v].get("weight", 1)
                    if dist[u] + w < dist[v]:
                        dist[v] = dist[u] + w
                        prev[v] = u
                        heapq.heappush(pq, (dist[v], v))

            if dist[end] == float("inf"):
                self.result_text.insert(tk.END, "Không có đường đi.\n")
                return

            path = []
            x = end
            while x is not None:
                path.append(x)
                x = prev[x]
            path.reverse()

            self.result_text.insert(
                tk.END,
                f"Đường đi ngắn nhất {start} -> {end}: {' -> '.join(path)} (chi phí = {dist[end]})\n"
            )

            edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            self.draw_graph(highlight_nodes=path, highlight_edges=edges)
            return

        # --- STEP-BY-STEP MODE ---
        self.result_text.insert(tk.END, "Dijkstra Animation...\n")
        self.draw_graph()

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue

            self.draw_graph(highlight_nodes=[u])
            self.root.update()
            time.sleep(0.5)

            if u == end:
                break

            for v in self.G.neighbors(u):
                w = self.G[u][v].get("weight", 1)
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u
                    heapq.heappush(pq, (dist[v], v))

                    self.draw_graph(highlight_nodes=[u, v], highlight_edges=[(u, v)])
                    self.root.update()
                    time.sleep(0.5)

        if dist[end] == float("inf"):
            self.result_text.insert(tk.END, "Không có đường đi.\n")
            return

        path = []
        x = end
        while x is not None:
            path.append(x)
            x = prev[x]
        path.reverse()

        self.result_text.insert(
            tk.END,
            f"Đường đi ngắn nhất {start} -> {end}: {' -> '.join(path)} (chi phí = {dist[end]})\n"
        )

        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        self.draw_graph(highlight_nodes=path, highlight_edges=edges)

    #=======================================================================
    # PRIM
    #=======================================================================
    def handle_prim(self, start):
        if self.G.number_of_nodes() == 0:
            return

        H = self.G.to_undirected()
        start = start if start in H else list(H.nodes())[0]

        visited = {start}
        heap = []
        for v in H.neighbors(start):
            w = H[start][v].get("weight", 1)
            heapq.heappush(heap, (w, start, v))

        mst = []
        total = 0

        while heap:
            w, u, v = heapq.heappop(heap)
            if v in visited: continue
            visited.add(v)
            mst.append((u, v))
            total += w

            for x in H.neighbors(v):
                if x not in visited:
                    w2 = H[v][x].get("weight", 1)
                    heapq.heappush(heap, (w2, v, x))

        self.result_text.insert(tk.END,
            f"MST Prim (tổng = {total}):\n" +
            ", ".join(f"({u},{v})" for u, v in mst) + "\n"
        )

        self.draw_graph(highlight_edges=mst)

    #=======================================================================
    # KRUSKAL
    #=======================================================================
    def handle_kruskal(self):
        H = self.G.to_undirected()
        edges = []
        for u, v, data in H.edges(data=True):
            edges.append((data.get("weight", 1), u, v))
        edges.sort()

        parent = {}
        rank = {}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb: return False
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1
            return True

        for n in H.nodes():
            parent[n] = n
            rank[n] = 0

        mst = []
        total = 0
        for w, u, v in edges:
            if union(u, v):
                mst.append((u, v))
                total += w

        self.result_text.insert(tk.END,
            f"MST Kruskal (tổng = {total}):\n" +
            ", ".join(f"({u},{v})" for u, v in mst) + "\n"
        )
        self.draw_graph(highlight_edges=mst)

    #=======================================================================
    # EULER
    #=======================================================================
    def handle_euler(self, start):
        H = self.G.to_undirected()

        odd = [n for n, d in H.degree() if d % 2 == 1]
        if len(odd) not in (0, 2):
            messagebox.showwarning("Lỗi", "Đồ thị không có chu trình/đường Euler.")
            return

        if start not in H:
            start = odd[0] if odd else list(H.nodes())[0]

        adj = {u: list(H.neighbors(u)) for u in H.nodes()}
        stack = [start]
        circuit = []

        while stack:
            u = stack[-1]
            if adj[u]:
                v = adj[u].pop()
                adj[v].remove(u)
                stack.append(v)
            else:
                circuit.append(stack.pop())

        circuit.reverse()

        edges = [(circuit[i], circuit[i+1]) for i in range(len(circuit)-1)]

        self.result_text.insert(tk.END,
            "Euler path/cycle:\n" + " -> ".join(circuit) + "\n"
        )

        self.draw_graph(highlight_edges=edges, highlight_nodes=circuit)

    #=======================================================================
    # MAX FLOW
    #=======================================================================
    def handle_max_flow(self, start, end):
        if start not in self.G or end not in self.G:
            messagebox.showwarning("Lỗi", "Start/End không hợp lệ.")
            return

        # tạo capacity
        nodes = list(self.G.nodes())
        cap = {u: {v:0 for v in nodes} for u in nodes}

        for u, v, data in self.G.edges(data=True):
            w = data.get("weight", 1)
            cap[u][v] += w
            if not self.is_directed:
                cap[v][u] += w

        def bfs(parent):
            visited = set()
            q = deque([start])
            visited.add(start)
            while q:
                u = q.popleft()
                for v in nodes:
                    if v not in visited and cap[u][v] > 0:
                        visited.add(v)
                        parent[v] = u
                        q.append(v)
                        if v == end:
                            return True
            return False

        parent = {}
        flow = 0

        while bfs(parent):
            bottleneck = float("inf")
            v = end
            while v != start:
                u = parent[v]
                bottleneck = min(bottleneck, cap[u][v])
                v = u

            flow += bottleneck

            v = end
            while v != start:
                u = parent[v]
                cap[u][v] -= bottleneck
                cap[v][u] += bottleneck
                v = u

        self.result_text.insert(tk.END, f"Max flow = {flow}\n")
        self.draw_graph()

    #=======================================================================
    # CHECK 2-WAY
    #=======================================================================
    def handle_two_way(self, start, end):
        if start not in self.G or end not in self.G:
            messagebox.showwarning("Lỗi", "Start/End không hợp lệ.")
            return

        def reachable(s, t):
            visited = set()
            q = deque([s])
            while q:
                u = q.popleft()
                if u == t: return True
                for v in self.G.neighbors(u):
                    if v not in visited:
                        visited.add(v)
                        q.append(v)
            return False

        r1 = reachable(start, end)
        r2 = reachable(end, start)

        if r1 and r2:
            msg = "Liên thông 2 chiều."
        elif r1:
            msg = f"{start} → {end} thông."
        elif r2:
            msg = f"{end} → {start} thông."
        else:
            msg = "Không có đường đi."

        self.result_text.insert(tk.END, msg + "\n")
        self.draw_graph(highlight_nodes=[start, end])

    #=======================================================================
    # CHUYỂN MA TRẬN / DANH SÁCH KỀ
    #=======================================================================
    def handle_convert_representation(self):
        nodes = list(self.G.nodes())
        idx = {n:i for i,n in enumerate(nodes)}
        n = len(nodes)

        matrix = [[0]*n for _ in range(n)]
        for u,v in self.G.edges():
            i, j = idx[u], idx[v]
            matrix[i][j] = 1
            if not self.is_directed:
                matrix[j][i] = 1

        self.result_text.insert(tk.END, "Ma trận kề:\n")
        for row in matrix:
            self.result_text.insert(tk.END, " ".join(map(str,row)) + "\n")

        self.result_text.insert(tk.END, "\nDanh sách kề:\n")
        for u in nodes:
            self.result_text.insert(tk.END, f"{u}: {list(self.G.neighbors(u))}\n")

        self.result_text.insert(tk.END, "\nDanh sách cạnh:\n")
        for u,v in self.G.edges():
            self.result_text.insert(tk.END, f"({u},{v})\n")


#====================== RUN APP ======================
if __name__ == "__main__":
    root = tk.Tk()
    app = GraphApp(root)
    root.mainloop()
